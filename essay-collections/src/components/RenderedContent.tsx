import { sanitizeEssayHtml } from "@/lib/sanitize";

/**
 * Renders stored essay HTML. Content is sanitized on the way IN (before store)
 * and again here on the way OUT — defense in depth against stored XSS.
 */
export function RenderedContent({ html }: { html: string }) {
  const clean = sanitizeEssayHtml(html);
  return <div className="essay" dangerouslySetInnerHTML={{ __html: clean }} />;
}
