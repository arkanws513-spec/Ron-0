import test from "node:test";
import assert from "node:assert/strict";
import "../ron-inference-client.js";

const client = globalThis.RonInferenceClient;

test("native client posts bounded recent context to the configured endpoint", async () => {
  let captured;
  const result = await client.chat(
    "https://ron.example.test///",
    [
      { role: "system", content: "rules" },
      { role: "user", content: "old question" },
      { role: "assistant", content: "old answer" },
      { role: "user", content: "ما عاصمة مصر؟" }
    ],
    async (url, options) => {
      captured = { url, options };
      return {
        ok: true,
        json: async () => ({
          content: "عاصمة مصر هي القاهرة.",
          model: "ron-native-checkpoint",
          metadata: { external_model_used: false }
        })
      };
    }
  );
  assert.equal(captured.url, "https://ron.example.test/v1/chat");
  assert.equal(captured.options.method, "POST");
  assert.deepEqual(JSON.parse(captured.options.body).messages.map(x => x.role), ["system", "user", "assistant", "user"]);
  assert.equal(result.content, "عاصمة مصر هي القاهرة.");
  assert.equal(result.metadata.external_model_used, false);
});

test("native client limits history and each message length", async () => {
  let body;
  const messages = Array.from({ length: 20 }, (_, i) => ({
    role: "user",
    content: String(i) + "x".repeat(5000)
  }));
  await client.chat("https://ron.example.test", messages, async (_url, options) => {
    body = JSON.parse(options.body);
    return { ok: true, json: async () => ({ content: "ok" }) };
  });
  assert.equal(body.messages.length, 12);
  assert.ok(body.messages.every(x => x.content.length <= 4000));
});

test("native client safely falls back when endpoint is unset or unavailable", async () => {
  let called = false;
  assert.equal(await client.chat("", [{ role: "user", content: "hello" }], async () => {
    called = true;
    return { ok: true, json: async () => ({ content: "not used" }) };
  }), null);
  assert.equal(called, false);
  assert.equal(await client.chat("https://ron.example.test", [{ role: "user", content: "hello" }], async () => ({ ok: false })), null);
});
