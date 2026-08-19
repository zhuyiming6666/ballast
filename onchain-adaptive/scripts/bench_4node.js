const fs = require("node:fs");
const path = require("node:path");
const { performance } = require("node:perf_hooks");
const { ethers, network } = require("hardhat");

async function timed(label, txPromise) {
  const start = performance.now();
  const tx = await txPromise;
  const receipt = await tx.wait();
  return { operation: label, latency_ms: performance.now() - start,
           gas: receipt.gasUsed.toString() };
}

async function main() {
  const [operator, alice, bob, carol] = await ethers.getSigners();
  const Factory = await ethers.getContractFactory("BallastAdaptive");
  const contract = await Factory.deploy(10);
  await contract.waitForDeployment();
  const rows = [];
  rows.push(await timed("postBond", contract.postBond(
    ethers.id("4node-sid"), ethers.id("root-0"), ethers.parseEther("3"),
    ethers.id("policy-m8-q1-r200ms"), { value: ethers.parseEther("12") })));
  rows.push(await timed("checkpoint", contract.checkpoint(1, ethers.id("root-1"))));

  let currentRoot = ethers.id("root-1");
  for (const [index, peer] of [alice, bob, carol].entries()) {
    const successorRoot = ethers.id(`confirmed-root-${index + 1}`);
    rows.push(await timed(`confirm-peer-${index + 1}`, contract.confirm(
      ethers.id(`draw-${index}`), ethers.parseEther("1"), currentRoot,
      successorRoot)));
    currentRoot = successorRoot;
    const claimId = ethers.id(`claim-${index}`);
    const channelId = ethers.id(`channel-${index}`);
    const amount = ethers.parseEther("1");
    const digest = await contract.claimDigest(
      claimId, channelId, peer.address, amount, 1, 0, index + 1,
      ethers.id("root-1"));
    const signature = await operator.signMessage(ethers.getBytes(digest));
    rows.push(await timed(`claim-peer-${index + 1}`, contract.connect(peer).openClaim(
      claimId, channelId, amount, 1, 0, index + 1,
      ethers.id("root-1"), signature)));
  }

  await network.provider.send("evm_increaseTime", [11]);
  await network.provider.send("evm_mine");
  for (let index = 0; index < 3; index++) {
    rows.push(await timed(`settle-peer-${index + 1}`,
      contract.settleClaim(ethers.id(`claim-${index}`))));
  }
  const bond = await contract.bond();
  if (bond.escrowed !== 0n || bond.remaining !== ethers.parseEther("9")) {
    throw new Error("four-node settlement invariant failed");
  }
  const target = path.join(__dirname, "..", "results", "four_node_e2e.csv");
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, "operation,latency_ms,gas\n" +
    rows.map(r => `${r.operation},${r.latency_ms.toFixed(3)},${r.gas}`).join("\n") + "\n");
  console.log(fs.readFileSync(target, "utf8"));
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
