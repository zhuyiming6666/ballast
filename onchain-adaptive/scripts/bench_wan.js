// bench_wan.js -- end-to-end operation latency against a node reached over a
// WAN-emulating delay proxy.  Run a `hardhat node` on 8545 and delay proxies
// on the ports given in RTTS, then: node scripts/bench_wan.js
//
// Also measures the off-chain fast path (sign + verify of a draw
// certificate), whose end-to-end latency is one bilateral RTT plus crypto.

const fs = require("node:fs");
const path = require("node:path");
const { performance } = require("node:perf_hooks");
const { ethers } = require("ethers");

// hardhat node default account #0/#1
const KEY0 = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
const KEY1 = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d";

const RTTS = { 0: 8545, 50: 8546, 100: 8547, 200: 8548 }; // rtt_ms -> port

const artifact = require(path.join(__dirname, "..", "artifacts", "contracts",
  "BallastAdaptive.sol", "BallastAdaptive.json"));

async function timed(fn) {
  const t0 = performance.now();
  const tx = await fn();
  await tx.wait();
  return performance.now() - t0;
}

async function benchAt(rtt, port, rows) {
  const provider = new ethers.JsonRpcProvider(`http://127.0.0.1:${port}`,
    undefined, { batchMaxCount: 1, polling: true, pollingInterval: 20 });
  const operator = new ethers.NonceManager(new ethers.Wallet(KEY0, provider));
  const alice = new ethers.NonceManager(new ethers.Wallet(KEY1, provider));
  const factory = new ethers.ContractFactory(
    artifact.abi, artifact.bytecode, operator);
  const contract = await factory.deploy(10);
  await contract.waitForDeployment();
  const sid = ethers.id(`wan-${rtt}`);
  const record = (op, ms) =>
    rows.push({ rtt_ms: rtt, operation: op, latency_ms: ms.toFixed(1) });

  record("postBond", await timed(() => contract.postBond(
    sid, ethers.id("root-0"), ethers.parseEther("2"),
    ethers.id("policy-m8-q1-r200ms"), { value: ethers.parseEther("10") })));
  record("checkpoint", await timed(() =>
    contract.checkpoint(1, ethers.id(`root-1-${rtt}`))));
  record("confirm_CAS", await timed(() => contract.confirm(
    ethers.id(`draw-${rtt}`), ethers.parseEther("1"),
    ethers.id(`root-1-${rtt}`), ethers.id(`root-2-${rtt}`))));

  const claimId = ethers.id(`claim-${rtt}`);
  const digest = await contract.claimDigest(
    claimId, ethers.id("ch-1"), await alice.getAddress(), ethers.parseEther("1"),
    1, 0, 1, ethers.id(`root-1-${rtt}`));
  const sig = await operator.signMessage(ethers.getBytes(digest));
  record("openClaim", await timed(() => contract.connect(alice).openClaim(
    claimId, ethers.id("ch-1"), ethers.parseEther("1"), 1, 0, 1,
    ethers.id(`root-1-${rtt}`), sig)));

  // off-chain fast path: certificate sign + local verify (no chain touch);
  // end-to-end = one bilateral RTT + this crypto time.
  const msg = ethers.getBytes(ethers.id("draw-certificate"));
  const t0 = performance.now();
  const N = 50;
  for (let i = 0; i < N; i++) {
    const s = await operator.signMessage(msg);
    ethers.verifyMessage(msg, s);
  }
  const crypto_ms = (performance.now() - t0) / N;
  record("draw_fastpath_offchain", rtt + crypto_ms);
  provider.destroy();
}

async function main() {
  const rows = [];
  for (const [rtt, port] of Object.entries(RTTS)) {
    await benchAt(Number(rtt), Number(port), rows);
    console.log("done rtt", rtt);
  }
  const target = path.join(__dirname, "..", "results", "wan_latency.csv");
  fs.writeFileSync(target, "rtt_ms,operation,latency_ms\n" +
    rows.map(r => `${r.rtt_ms},${r.operation},${r.latency_ms}`).join("\n") + "\n");
  console.log(fs.readFileSync(target, "utf8"));
}

main().catch(e => { console.error(e); process.exitCode = 1; });
