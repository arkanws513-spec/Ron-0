import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const context = { console };
vm.runInNewContext(fs.readFileSync("ron-browser-reasoner.js","utf8"), context);
const R = context.RonReasoner;

assert.ok(R);
assert.equal(R.parse("عاصمة مصر هي القاهرة").subject, "القاهرة");
assert.equal(R.parse("عاصمة مصر هي القاهرة").object, "مصر");

const facts = R.parseAll([
  "عاصمة مصر هي القاهرة",
  "القاهرة تقع في مصر",
  "الليل يأتي بعد النهار"
]);

const capital = R.answer("ما هي عاصمة مصر", facts);
assert.equal(capital.answer, "القاهرة هي عاصمة مصر.");

const temporal = R.answer("ما الذي يأتي بعد النهار", facts);
assert.ok(temporal && temporal.answer.includes("بعد"));

const chained = R.reason(R.parseAll([
  "أ يتبع ب",
  "ب يتبع ج"
]));
assert.ok(chained.some(x => x.relation === "follows" && x.subject === "ا" && x.object === "ج"));

console.log("reasoner runtime smoke test passed");
