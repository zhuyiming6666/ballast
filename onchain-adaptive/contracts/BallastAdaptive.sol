// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title BALLAST transparent fork-containment prototype
/// @notice Implements the on-chain pieces of Pi(tau_e, theta): finalized
/// checkpoint anchors, a hard-isolated safety reserve, CAS confirmation for
/// the strict endpoint, and amount-scoped claim escrow. Bilateral fast-path
/// certificates remain off-chain. This is a research prototype, not a full
/// payment-channel integration.
contract BallastAdaptive {
    uint256 private constant SECP256K1N_HALF =
        0x7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a0;
    bytes32 public constant CLAIM_TYPEHASH = keccak256(
        "BallastClaim(uint256 chainId,address contractAddress,bytes32 sid,bytes32 claimId,bytes32 channelId,address claimant,uint256 amount,uint64 checkpointEpoch,uint64 modeId,uint64 slot,bytes32 checkpointRoot)"
    );

    struct Bond {
        address operator;
        bytes32 sid;
        uint256 remaining;
        uint256 safetyReserve;
        uint256 escrowed;
        uint256 confirmedOutstanding;
        uint64 epoch;
        uint64 modeId;
        bytes32 checkpointRoot;
        bytes32 currentRoot;
        bytes32 admissionPolicyId;
        bool live;
    }

    struct Claim {
        address payable claimant;
        uint256 amount;
        uint64 deadline;
        bool open;
    }

    uint64 public immutable DELTA;
    Bond public bond;
    uint64 public releaseAt;
    mapping(bytes32 => Claim) public claims;
    mapping(bytes32 => uint256) public confirmedDrawAmounts;
    mapping(bytes32 => bool) public seenDrawIds;
    mapping(bytes32 => bool) public seenClaimSlots;
    mapping(uint64 => bytes32) public sealedRoots;
    mapping(uint64 => uint64) public checkpointModes;
    mapping(uint64 => bytes32) public modePolicies;
    mapping(uint64 => uint256) public modeSafetyReserves;

    event BondPosted(
        bytes32 indexed sid, address indexed operator, uint256 amount,
        uint256 safetyReserve, bytes32 admissionPolicyId
    );
    event BondToppedUp(uint256 amount, uint256 remaining);
    event Checkpointed(uint64 indexed epoch, bytes32 indexed root, uint64 modeId);
    event ModeUpdated(
        uint64 indexed modeId, bytes32 indexed admissionPolicyId,
        uint256 safetyReserve
    );
    event Confirmed(
        bytes32 indexed drawId, bytes32 indexed successorRoot, uint256 amount
    );
    event ConfirmRetired(bytes32 indexed drawId, uint256 amount);
    event ClaimEscrowed(
        bytes32 indexed claimId, bytes32 indexed channelId,
        address indexed claimant, uint256 amount, uint64 checkpointEpoch,
        uint64 modeId, uint64 slot
    );
    event ClaimSettled(
        bytes32 indexed claimId, address indexed claimant, uint256 amount
    );

    constructor(uint64 delta_) {
        require(delta_ > 0, "zero delta");
        DELTA = delta_;
    }

    function postBond(
        bytes32 sid,
        bytes32 initialRoot,
        uint256 safetyReserve,
        bytes32 admissionPolicyId
    ) external payable {
        require(!bond.live, "live");
        require(
            msg.value > safetyReserve && sid != bytes32(0)
                && initialRoot != bytes32(0)
                && admissionPolicyId != bytes32(0),
            "invalid setup"
        );
        bond = Bond({
            operator: msg.sender,
            sid: sid,
            remaining: msg.value,
            safetyReserve: safetyReserve,
            escrowed: 0,
            confirmedOutstanding: 0,
            epoch: 0,
            modeId: 0,
            checkpointRoot: initialRoot,
            currentRoot: initialRoot,
            admissionPolicyId: admissionPolicyId,
            live: true
        });
        sealedRoots[0] = initialRoot;
        checkpointModes[0] = 0;
        modePolicies[0] = admissionPolicyId;
        modeSafetyReserves[0] = safetyReserve;
        releaseAt = 0;
        emit BondPosted(
            sid, msg.sender, msg.value, safetyReserve, admissionPolicyId
        );
        emit Checkpointed(0, initialRoot, 0);
    }

    function topUp() external payable {
        require(msg.sender == bond.operator && bond.live, "operator");
        require(msg.value > 0, "zero");
        bond.remaining += msg.value;
        releaseAt = 0;
        emit BondToppedUp(msg.value, bond.remaining);
    }

    /// @notice Public seal under the active mode. Recipients enforce the
    /// tau_e + delta freshness rule before honor; the contract retains every
    /// historical seal so a later claim does not depend on current freshness.
    function checkpoint(uint64 nextEpoch, bytes32 nextRoot) external {
        require(msg.sender == bond.operator && bond.live, "operator");
        _checkpoint(nextEpoch, nextRoot);
    }

    /// @notice Installs a fresh mode and checkpoint atomically. A mode whose
    /// reserve does not fit current liabilities must wait for top-up/retirement.
    function updateMode(
        uint256 newSafetyReserve,
        bytes32 newAdmissionPolicyId,
        uint64 nextEpoch,
        bytes32 nextRoot
    ) external {
        require(msg.sender == bond.operator && bond.live, "operator");
        require(newAdmissionPolicyId != bytes32(0), "zero policy");
        require(
            newSafetyReserve + bond.escrowed + bond.confirmedOutstanding
                <= bond.remaining,
            "reserve underfunded"
        );
        uint64 nextMode = bond.modeId + 1;
        bond.modeId = nextMode;
        bond.safetyReserve = newSafetyReserve;
        bond.admissionPolicyId = newAdmissionPolicyId;
        modePolicies[nextMode] = newAdmissionPolicyId;
        modeSafetyReserves[nextMode] = newSafetyReserve;
        emit ModeUpdated(nextMode, newAdmissionPolicyId, newSafetyReserve);
        _checkpoint(nextEpoch, nextRoot);
    }

    function _checkpoint(uint64 nextEpoch, bytes32 nextRoot) private {
        require(
            nextEpoch == bond.epoch + 1 && nextRoot != bytes32(0),
            "epoch/root"
        );
        bond.epoch = nextEpoch;
        bond.checkpointRoot = nextRoot;
        bond.currentRoot = nextRoot;
        sealedRoots[nextEpoch] = nextRoot;
        checkpointModes[nextEpoch] = bond.modeId;
        releaseAt = 0;
        emit Checkpointed(nextEpoch, nextRoot, bond.modeId);
    }

    /// @notice Strong endpoint: contract storage is a parent-to-successor CAS.
    /// Two transitions naming the same expected parent cannot both confirm.
    function confirm(
        bytes32 drawId,
        uint256 amount,
        bytes32 expectedParent,
        bytes32 successorRoot
    ) external {
        require(msg.sender == bond.operator && bond.live, "operator");
        require(!seenDrawIds[drawId] && amount > 0, "duplicate/zero");
        require(expectedParent == bond.currentRoot, "stale parent");
        require(
            successorRoot != bytes32(0) && successorRoot != expectedParent,
            "successor"
        );
        require(amount <= available(), "ceiling");
        seenDrawIds[drawId] = true;
        confirmedDrawAmounts[drawId] = amount;
        bond.confirmedOutstanding += amount;
        bond.currentRoot = successorRoot;
        releaseAt = 0;
        emit Confirmed(drawId, successorRoot, amount);
    }

    function retireConfirmed(bytes32 drawId, uint256 amount) external {
        require(msg.sender == bond.operator, "operator");
        require(
            amount > 0 && amount == confirmedDrawAmounts[drawId],
            "draw/amount"
        );
        delete confirmedDrawAmounts[drawId];
        bond.confirmedOutstanding -= amount;
        emit ConfirmRetired(drawId, amount);
    }

    /// @notice Opens an ordinary honored-draw claim against service headroom.
    /// The signed evidence binds its historical checkpoint, mode, and
    /// recipient-specific slot. No current-freshness check is performed.
    function openClaim(
        bytes32 claimId,
        bytes32 channelId,
        uint256 amount,
        uint64 checkpointEpoch,
        uint64 modeId,
        uint64 slot,
        bytes32 checkpointRoot,
        bytes calldata operatorSig
    ) external {
        require(
            bond.live && claims[claimId].claimant == address(0),
            "dead/duplicate"
        );
        require(amount > 0 && checkpointRoot != bytes32(0), "zero");
        require(
            sealedRoots[checkpointEpoch] == checkpointRoot
                && checkpointModes[checkpointEpoch] == modeId
                && modePolicies[modeId] != bytes32(0),
            "historical checkpoint"
        );
        bytes32 slotKey = claimSlotKey(
            channelId, msg.sender, checkpointEpoch, modeId, slot
        );
        require(!seenClaimSlots[slotKey], "slot replay");
        bytes32 digest = claimDigest(
            claimId, channelId, msg.sender, amount, checkpointEpoch, modeId,
            slot, checkpointRoot
        );
        require(_recover(digest, operatorSig) == bond.operator, "signature");
        require(amount <= available(), "ceiling");
        seenClaimSlots[slotKey] = true;
        bond.escrowed += amount;
        claims[claimId] = Claim(
            payable(msg.sender), amount, uint64(block.timestamp) + DELTA, true
        );
        releaseAt = 0;
        emit ClaimEscrowed(
            claimId, channelId, msg.sender, amount, checkpointEpoch, modeId,
            slot
        );
    }

    function settleClaim(bytes32 claimId) external {
        Claim storage claim_ = claims[claimId];
        require(claim_.open && block.timestamp >= claim_.deadline, "window");
        claim_.open = false;
        uint256 amount = claim_.amount;
        bond.escrowed -= amount;
        bond.remaining -= amount;
        (bool ok,) = claim_.claimant.call{value: amount}("");
        require(ok, "transfer");
        emit ClaimSettled(claimId, claim_.claimant, amount);
    }

    /// @notice Service headroom. The safety reserve is never available to an
    /// ordinary confirmation or claim.
    function available() public view returns (uint256) {
        return bond.remaining - bond.safetyReserve - bond.escrowed
            - bond.confirmedOutstanding;
    }

    function requestRelease() external {
        require(msg.sender == bond.operator && bond.live, "operator");
        require(
            bond.escrowed == 0 && bond.confirmedOutstanding == 0,
            "not zero"
        );
        releaseAt = uint64(block.timestamp) + DELTA;
    }

    function finalizeRelease() external {
        require(msg.sender == bond.operator && bond.live, "operator");
        require(releaseAt != 0 && block.timestamp >= releaseAt, "window");
        require(
            bond.escrowed == 0 && bond.confirmedOutstanding == 0,
            "not zero"
        );
        bond.live = false;
        uint256 amount = bond.remaining;
        bond.remaining = 0;
        (bool ok,) = payable(bond.operator).call{value: amount}("");
        require(ok, "transfer");
    }

    function claimSlotKey(
        bytes32 channelId,
        address claimant,
        uint64 checkpointEpoch,
        uint64 modeId,
        uint64 slot
    ) public view returns (bytes32) {
        return keccak256(abi.encode(
            bond.sid, modeId, checkpointEpoch, slot, channelId, claimant
        ));
    }

    function claimDigest(
        bytes32 claimId,
        bytes32 channelId,
        address claimant,
        uint256 amount,
        uint64 checkpointEpoch,
        uint64 modeId,
        uint64 slot,
        bytes32 checkpointRoot
    ) public view returns (bytes32) {
        require(claimant != address(0), "zero claimant");
        return keccak256(abi.encode(
            CLAIM_TYPEHASH, block.chainid, address(this), bond.sid,
            claimId, channelId, claimant, amount, checkpointEpoch, modeId,
            slot, checkpointRoot
        ));
    }

    function _recover(
        bytes32 digest,
        bytes calldata signature
    ) private pure returns (address) {
        require(signature.length == 65, "sig length");
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := calldataload(signature.offset)
            s := calldataload(add(signature.offset, 32))
            v := byte(0, calldataload(add(signature.offset, 64)))
        }
        if (v < 27) v += 27;
        require(
            (v == 27 || v == 28) && uint256(s) <= SECP256K1N_HALF,
            "sig"
        );
        bytes32 signedDigest = keccak256(
            abi.encodePacked("\x19Ethereum Signed Message:\n32", digest)
        );
        return ecrecover(signedDigest, v, r, s);
    }
}
