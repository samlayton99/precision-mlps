import "server-only";
import Anthropic from "@anthropic-ai/sdk";

/**
 * The rubric the daily bot judges against — a compact projection of the
 * community guidelines. Keep this in sync with docs/COMMUNITY_GUIDELINES.md
 * and src/app/guidelines/page.tsx.
 */
export const MODERATION_RUBRIC = `You are a careful, charitable content-safety reviewer for an essay forum
for members of The Church of Jesus Christ of Latter-day Saints.
The community's purpose is thoughtful, faithful writing that helps people come
closer to Jesus Christ and build Zion. Hard, sincere questions asked in good
faith are WELCOME and must NOT be flagged.

Flag content only if it clearly violates the guidelines:
- Hateful, demeaning, harassing, or threatening speech toward any person or group.
- Content that attacks or seeks to undermine the Church, its leaders, or its
  doctrine (apostate / "anti-" content). (Sincere questions are NOT this.)
- Profanity, obscenity, or sexually explicit / graphic material.
- Personal attacks, doxxing (sharing private info), or spam / advertising.

Err on the side of NOT flagging. Faithful disagreement, doubt expressed in good
faith, and difficult doctrinal or historical questions are allowed.`;

export interface ModerationItem {
  id: string;
  type: "post" | "comment" | "chat" | "profile";
  text: string;
}

export interface ModerationVerdict {
  id: string;
  violation: boolean;
  category: string | null;
  severity: "low" | "medium" | "high" | null;
  reason: string;
}

/**
 * Ask Claude to judge a batch of items. Returns one verdict per item. Robust to
 * SDK/model output variation: we request strict JSON and parse defensively.
 */
export async function judgeBatch(
  client: Anthropic,
  model: string,
  items: ModerationItem[],
): Promise<ModerationVerdict[]> {
  const payload = items.map((i) => ({
    id: i.id,
    type: i.type,
    text: i.text.slice(0, 4000),
  }));

  const prompt = `${MODERATION_RUBRIC}

Review each item below. Return ONLY a JSON array (no prose, no code fences) with
one object per item, in the same order, shaped exactly:
{"id": string, "violation": boolean, "category": string|null, "severity": "low"|"medium"|"high"|null, "reason": string}
Set violation=false for anything acceptable, with reason="" and category/severity null.

ITEMS:
${JSON.stringify(payload)}`;

  const res = await client.messages.create({
    model,
    max_tokens: 2000,
    messages: [{ role: "user", content: prompt }],
  });

  const text = res.content
    .map((b) => (b.type === "text" ? b.text : ""))
    .join("")
    .trim();

  const parsed = extractJsonArray(text);
  if (!parsed) return [];

  return parsed
    .filter((v): v is Record<string, unknown> => typeof v === "object" && v !== null)
    .map((v): ModerationVerdict => {
      const sev = v.severity;
      return {
        id: String(v.id ?? ""),
        violation: Boolean(v.violation),
        category: v.category ? String(v.category) : null,
        severity: sev === "low" || sev === "medium" || sev === "high" ? sev : null,
        reason: v.reason ? String(v.reason) : "",
      };
    })
    .filter((v) => v.id);
}

/** Pull the first JSON array out of a model response, tolerating stray text/fences. */
function extractJsonArray(text: string): unknown[] | null {
  const cleaned = text.replace(/```json\s*/gi, "").replace(/```/g, "");
  const start = cleaned.indexOf("[");
  const end = cleaned.lastIndexOf("]");
  if (start === -1 || end === -1 || end < start) return null;
  try {
    const val = JSON.parse(cleaned.slice(start, end + 1));
    return Array.isArray(val) ? val : null;
  } catch {
    return null;
  }
}
