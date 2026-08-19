// bench_sepolia.js -- real-testnet measurement: gas + end-to-end inclusion
// latency for the BALLAST contract operations on Sepolia.
//
//   SEPOLIA_RPC (default https://ethereum-sepolia-rpc.publicnode.com)
//   key read from ~/.ballast_sepolia_key
//   node scripts/bench_sepolia.js [rounds]
//
// Deploys once, then measures postBond, `rounds` x (checkpoint, confirm),
// one openClaim + settleClaim.  Latency = submit -> first confirmation.

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { performance } = require("node:perf_hooks");
const { ethers } = require("ethers");

const RPC = process.env.SEPOLIA_RPC ||
  "https://ethereum-sepolia-rpc.publicnode.com";
const KEY = fs.readFileSync(path.join(os.homedir(), ".ballast_sepolia_key"),
  "utf8").trim();
const artifact = require(path.join(__dirname, "..", "artifacts", "contracts",
  "BallastAdaptive.sol", "BallastAdaptive.json"));

const rows = [];
async function timed(op, fn) {
  const t0 = performance.now();
  const tx = await fn();
  const sent = performance.now();
  const rc = await tx.wait(1);
  rows.push({ operation: op, gas: rc.gasUsed.toString(),
              submit_ms: (sent - t0).toFixed(0),
              inclusion_ms: (performance.now() - sent).toFixed(0),
              block: rc.blockNumber, txhash: rc.hash });
  console.log(op, "gas", rc.gasUsed.toString(), "incl",
              ((performance.now() - sent) / 1000).toFixed(1), "s");
  return rc;
}

async function main() {
  const rounds = Number(process.argv[2] || 10);
  const provider = new ethers.JsonRpcProvider(RPC);
  const operator = new ethers.NonceManager(new ethers.Wallet(KEY, provider));
  const addr = await operator.getAddress();
  const bal = await provider.getBalance(addr);
  console.log("account", addr, "balance", ethers.formatEther(bal), "ETH");
  if (bal < ethers.parseEther("0.02")) {
    console.error("insufficient Sepolia ETH; fund the address first");
    process.exit(1);
  }

  const factory = new ethers.ContractFactory(
    artifact.abi, artifact.bytecode, operator);
  const contract = await factory.deploy(10);
  await contract.waitForDeployment();
  console.log("deployed at", await contract.getAddress());

  const wei = (n) => ethers.parseUnits(n, "gwei");
  await timed("postBond", () => contract.postBond(
    ethers.id("sepolia-sid"), ethers.id("root-0"), wei("2"),
    ethers.id("policy-m8-q1-r200ms"), { value: wei("100") }));

  let root = ethers.id("root-0");
  for (let i = 1; i <= rounds; i++) {
    const next = ethers.id(`root-${i}`);
    await timed("checkpoint", () => contract.checkpoint(i, next));
    const conf = ethers.id(`confirmed-${i}`);
    await timed("confirm_CAS", () => contract.confirm(
      ethers.id(`draw-${i}`), wei("1"), next, conf));
    root = conf;
  }

  const claimId = ethers.id("sepolia-claim");
  const digest = await contract.claimDigest(
    claimId, ethers.id("ch-1"), addr, wei("1"), 1, 0, 1,
    ethers.id("root-1"));
  const sig = await operator.signer.signMessage(ethers.getBytes(digest));
  await timed("openClaim", () => contract.openClaim(
    claimId, ethers.id("ch-1"), wei("1"), 1, 0, 1,
    ethers.id("root-1"), sig));
  console.log("waiting out the challenge window (contract param)...");
  await new Promise(r => setTimeout(r, 15000));
  try {
    await timed("settleClaim", () => contract.settleClaim(claimId));
  } catch (e) {
    console.log("settleClaim skipped:", e.shortMessage || e.message);
  }

  const target = path.join(__dirname, "..", "results", "sepolia_latency.csv");
  fs.writeFileSync(target,
    "operation,gas,submit_ms,inclusion_ms,block,txhash\n" +
    rows.map(r => [r.operation, r.gas, r.submit_ms, r.inclusion_ms,
                   r.block, r.txhash].join(",")).join("\n") + "\n");
  console.log("wrote", target);
}

main().catch(e => { console.error(e); process.exitCode = 1; });
