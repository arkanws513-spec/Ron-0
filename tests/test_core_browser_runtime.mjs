import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const storage=new Map();
const context={
 console,
 localStorage:{
  getItem:k=>storage.get(k)??null,
  setItem:(k,v)=>storage.set(k,v),
  removeItem:k=>storage.delete(k)
 },
 Date, JSON, Math, setTimeout, clearTimeout, AbortController
};
vm.runInNewContext(fs.readFileSync("ron-core.js","utf8"),context);
assert.ok(context.RonCore);
assert.ok(context.RonCoreAdapters?.LocalStorageAdapter);
const ron=new context.RonCore({adapter:new context.RonCoreAdapters.LocalStorageAdapter("test-core")});
assert.match((await ron.handle("ما اسمك")).reply,/رون/);
assert.match((await ron.handle("اسمي زيريوس")).reply,/زيريوس/);
assert.match((await ron.handle("ما اسمي")).reply,/زيريوس/);
assert.match((await ron.handle("ما عاصمة مصر")).reply,/القاهرة/);
console.log("browser Ron Core runtime smoke test passed");
