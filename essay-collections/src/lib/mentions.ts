import { htmlToText } from "@/lib/sanitize";

/**
 * Extract unique @handles from text or HTML. Handles are 3–24 chars of
 * [a-z0-9_]. Used to turn "tag other users" into mention + notification rows.
 */
export function extractHandles(input: string, isHtml = false): string[] {
  const text = isHtml ? htmlToText(input) : input;
  const found = new Set<string>();
  const re = /(^|[^a-zA-Z0-9_])@([a-z0-9_]{3,24})/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    found.add(m[2]!.toLowerCase());
  }
  return [...found].slice(0, 25); // cap to avoid notification spam
}
