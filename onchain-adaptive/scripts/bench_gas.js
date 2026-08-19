const fs = require("node:fs");
const path = require("node:path");
const { ethers, network } = require("hardhat");

async function used(txPromise) {
  const tx = await txPromise;
  return (await tx.wait()).gasUsed;
}

async function main() {
  const [operator, alice] = await ethers.getSigners();
  const Factory = await ethers.getContractFactory("BallastAdaptive");
  const contract = await Factory.deploy(10);
  await contract.waitForDeployment();
  const rows = [];
  rows.push(["postBond", "once/operator", await used(contract.postBond(
    ethers.id("bench-sid"), ethers.id("root-0"), ethers.parseEther("2"),
    ethers.id("policy-m8-q1-r200ms"), { value: ethers.parseEther("10") }
  ))]);
  rows.push(["checkpoint", "per epoch", await used(contract.checkpoint(1, ethers.id("root-1")))]);
  rows.push(["confirm", "per draw > theta", await used(contract.confirm(
    ethers.id("draw-1"), ethers.parseEther("1"), ethers.id("root-1"),
    ethers.id("confirmed-root-1")
  ))]);
  const claimId = ethers.id("claim-1");
  const channelId = ethers.id("channel-1");
  const checkpointRoot = ethers.id("root-1");
  const amount = ethers.parseEther("2");
  const digest = await contract.claimDigest(
    claimId, channelId, alice.address, amount, 1, 0, 1, checkpointRoot
  );
  const signature = await operator.signMessage(ethers.getBytes(digest));
  rows.push(["openClaimEscrow", "per dispute", await used(contract.connect(alice).openClaim(
    claimId, channelId, amount, 1, 0, 1, checkpointRoot, signature
  ))]);
  await network.provider.send("evm_increaseTime", [11]);
  await network.provider.send("evm_mine");
  rows.push(["settleClaim", "per dispute", await used(contract.settleClaim(claimId))]);

  const target = path.join(__dirname, "..", "results", "adaptive_gas.csv");
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, "operation,frequency,gas\n" +
    rows.map(([a,b,c]) => `${a},${b},${c}`).join("\n") + "\n");
  console.log(fs.readFileSync(target, "utf8"));
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
