const assert = require("node:assert/strict");
const { ethers, network } = require("hardhat");

async function expectRevert(promise, message) {
  try { await promise; assert.fail("expected revert"); }
  catch (error) { assert.match(String(error), new RegExp(message)); }
}

describe("BallastAdaptive", function () {
  const ROOT0 = ethers.id("root-0");
  const POLICY0 = ethers.id("policy-m8-q1-r200ms");

  async function fixture() {
    const [operator, alice, bob] = await ethers.getSigners();
    const Factory = await ethers.getContractFactory("BallastAdaptive");
    const contract = await Factory.deploy(10);
    await contract.waitForDeployment();
    await contract.postBond(
      ethers.id("sid"), ROOT0, ethers.parseEther("2"), POLICY0,
      { value: ethers.parseEther("10") }
    );
    return { contract, operator, alice, bob };
  }

  async function claimEvidence(
    contract, operator, claimant, tag, amount, slot,
    checkpointEpoch = 0n, modeId = 0n, checkpointRoot = ROOT0,
    channelTag = tag
  ) {
    const claimId = ethers.id(`claim-${tag}`);
    const channelId = ethers.id(`channel-${channelTag}`);
    const digest = await contract.claimDigest(
      claimId, channelId, claimant.address, amount, checkpointEpoch,
      modeId, slot, checkpointRoot
    );
    const signature = await operator.signMessage(ethers.getBytes(digest));
    return {
      claimId, channelId, amount, checkpointEpoch, modeId, slot,
      checkpointRoot, signature
    };
  }

  async function openEvidence(contract, claimant, evidence) {
    return contract.connect(claimant).openClaim(
      evidence.claimId, evidence.channelId, evidence.amount,
      evidence.checkpointEpoch, evidence.modeId, evidence.slot,
      evidence.checkpointRoot, evidence.signature
    );
  }

  async function signedClaim(
    contract, operator, claimant, tag, amount, slot,
    checkpointEpoch = 0n, modeId = 0n, checkpointRoot = ROOT0,
    channelTag = tag
  ) {
    const evidence = await claimEvidence(
      contract, operator, claimant, tag, amount, slot, checkpointEpoch,
      modeId, checkpointRoot, channelTag
    );
    return openEvidence(contract, claimant, evidence);
  }

  it("hard-isolates the safety reserve from ordinary service", async function () {
    const { contract, operator, alice, bob } = await fixture();
    assert.equal(await contract.available(), ethers.parseEther("8"));
    await signedClaim(
      contract, operator, alice, "a", ethers.parseEther("4"), 1
    );
    await contract.confirm(
      ethers.id("draw"), ethers.parseEther("4"), ROOT0,
      ethers.id("root-confirmed")
    );
    assert.equal(await contract.available(), 0n);
    await expectRevert(
      signedClaim(contract, operator, bob, "b", 1n, 2), "ceiling"
    );
    assert.equal((await contract.bond()).safetyReserve, ethers.parseEther("2"));
  });

  it("escrows a claim without freezing unrelated checkpoints", async function () {
    const { contract, operator, alice } = await fixture();
    await signedClaim(
      contract, operator, alice, "a", ethers.parseEther("2"), 1
    );
    assert.equal((await contract.bond()).escrowed, ethers.parseEther("2"));
    await contract.checkpoint(1, ethers.id("root-1"));
    assert.equal((await contract.bond()).epoch, 1n);
    assert.equal(await contract.available(), ethers.parseEther("6"));
  });

  it("settles exactly the escrowed amount without reopening spent headroom", async function () {
    const { contract, operator, alice } = await fixture();
    const amount = ethers.parseEther("3");
    await signedClaim(contract, operator, alice, "a", amount, 1);
    const before = await contract.available();
    await network.provider.send("evm_increaseTime", [11]);
    await network.provider.send("evm_mine");
    await contract.settleClaim(ethers.id("claim-a"));
    const bond = await contract.bond();
    assert.equal(bond.escrowed, 0n);
    assert.equal(bond.remaining, ethers.parseEther("7"));
    assert.equal(await contract.available(), before);
  });

  it("rejects a second strong-endpoint successor for the same parent", async function () {
    const { contract } = await fixture();
    await contract.confirm(
      ethers.id("draw-a"), ethers.parseEther("1"), ROOT0,
      ethers.id("root-a")
    );
    await expectRevert(
      contract.confirm(
        ethers.id("draw-b"), ethers.parseEther("1"), ROOT0,
        ethers.id("root-b")
      ),
      "stale parent"
    );
  });

  it("retires only the amount recorded for the named draw", async function () {
    const { contract } = await fixture();
    const first = ethers.id("draw-first");
    const second = ethers.id("draw-second");
    const root1 = ethers.id("confirmed-1");
    await contract.confirm(first, ethers.parseEther("1"), ROOT0, root1);
    await contract.confirm(
      second, ethers.parseEther("2"), root1, ethers.id("confirmed-2")
    );
    await expectRevert(
      contract.retireConfirmed(first, ethers.parseEther("3")), "draw/amount"
    );
    await contract.retireConfirmed(first, ethers.parseEther("1"));
    assert.equal(
      (await contract.bond()).confirmedOutstanding, ethers.parseEther("2")
    );
    await expectRevert(
      contract.confirm(
        first, ethers.parseEther("1"), ethers.id("confirmed-2"),
        ethers.id("confirmed-3")
      ),
      "duplicate/zero"
    );
  });

  it("binds historical claim evidence to its intended claimant", async function () {
    const { contract, operator, alice, bob } = await fixture();
    const evidence = await claimEvidence(
      contract, operator, alice, "bound", ethers.parseEther("1"), 7
    );
    await expectRevert(
      openEvidence(contract, bob, evidence), "signature"
    );
    await openEvidence(contract, alice, evidence);
  });

  it("rejects reuse of a recipient-bound epoch slot", async function () {
    const { contract, operator, alice } = await fixture();
    await signedClaim(
      contract, operator, alice, "slot-a", ethers.parseEther("1"), 9,
      0n, 0n, ROOT0, "shared"
    );
    await expectRevert(
      signedClaim(
        contract, operator, alice, "slot-b", ethers.parseEther("1"), 9,
        0n, 0n, ROOT0, "shared"
      ),
      "slot replay"
    );
  });

  it("accepts historically fresh evidence after newer checkpoints", async function () {
    const { contract, operator, alice } = await fixture();
    const evidence = await claimEvidence(
      contract, operator, alice, "historical", ethers.parseEther("1"), 11
    );
    await contract.checkpoint(1, ethers.id("root-1"));
    await contract.checkpoint(2, ethers.id("root-2"));
    await openEvidence(contract, alice, evidence);
    assert.equal((await contract.bond()).escrowed, ethers.parseEther("1"));
  });

  it("rejects evidence not bound to a sealed historical checkpoint", async function () {
    const { contract, operator, alice } = await fixture();
    const evidence = await claimEvidence(
      contract, operator, alice, "stale", ethers.parseEther("1"), 12,
      0n, 0n, ethers.id("not-sealed")
    );
    await expectRevert(
      openEvidence(contract, alice, evidence), "historical checkpoint"
    );
  });

  it("requires top-up before installing an underfunded mode reserve", async function () {
    const { contract, operator, alice } = await fixture();
    await signedClaim(
      contract, operator, alice, "mode", ethers.parseEther("4"), 13
    );
    await contract.confirm(
      ethers.id("mode-draw"), ethers.parseEther("2"), ROOT0,
      ethers.id("mode-confirmed")
    );
    await expectRevert(
      contract.updateMode(
        ethers.parseEther("5"), ethers.id("policy-stricter"), 1,
        ethers.id("mode-root-1")
      ),
      "reserve underfunded"
    );
    await contract.topUp({ value: ethers.parseEther("1") });
    await contract.updateMode(
      ethers.parseEther("5"), ethers.id("policy-stricter"), 1,
      ethers.id("mode-root-1")
    );
    assert.equal((await contract.bond()).modeId, 1n);
    assert.equal(await contract.available(), 0n);
  });

  it("rejects replay of a settled claim", async function () {
    const { contract, operator, alice } = await fixture();
    const amount = ethers.parseEther("1");
    await signedClaim(contract, operator, alice, "replay", amount, 20);
    await network.provider.send("evm_increaseTime", [11]);
    await network.provider.send("evm_mine");
    await contract.settleClaim(ethers.id("claim-replay"));
    await expectRevert(
      signedClaim(contract, operator, alice, "replay", amount, 20),
      "dead/duplicate"
    );
  });

  it("cancels a pending release when new session activity occurs", async function () {
    const { contract } = await fixture();
    await contract.requestRelease();
    await contract.confirm(
      ethers.id("draw-after-release"), ethers.parseEther("1"), ROOT0,
      ethers.id("release-root")
    );
    await contract.retireConfirmed(
      ethers.id("draw-after-release"), ethers.parseEther("1")
    );
    await network.provider.send("evm_increaseTime", [11]);
    await network.provider.send("evm_mine");
    await expectRevert(contract.finalizeRelease(), "window");

    await contract.requestRelease();
    await contract.checkpoint(1, ethers.id("root-1"));
    await network.provider.send("evm_increaseTime", [11]);
    await network.provider.send("evm_mine");
    await expectRevert(contract.finalizeRelease(), "window");
  });
});
