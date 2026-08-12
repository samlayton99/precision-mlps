import DOMPurify from "isomorphic-dompurify";

/**
 * Sanitize essay HTML before it is stored AND again before it is rendered
 * (defense in depth against stored XSS). We allow a Substack-like set of
 * formatting tags plus YouTube iframes, and nothing else. Any <script>, event
 * handler, javascript: URL, or non-YouTube iframe is stripped.
 */
const ALLOWED_TAGS = [
  "p", "br", "hr", "h1", "h2", "h3", "h4",
  "strong", "b", "em", "i", "u", "s", "del",
  "blockquote", "ul", "ol", "li",
  "a", "img", "figure", "figcaption",
  "pre", "code", "span", "iframe",
];

const ALLOWED_ATTR = [
  "href", "target", "rel", "src", "alt", "title",
  "width", "height", "class",
  "allow", "allowfullscreen", "frameborder", "loading",
  "data-youtube-video",
];

const YOUTUBE_HOSTS = new Set([
  "www.youtube.com",
  "youtube.com",
  "www.youtube-nocookie.com",
  "youtube-nocookie.com",
]);

// Restrict iframes to YouTube, and force external links to be safe.
DOMPurify.addHook("uponSanitizeElement", (node, data) => {
  if (data.tagName === "iframe") {
    const src = (node as Element).getAttribute("src") || "";
    let ok = false;
    try {
      ok = YOUTUBE_HOSTS.has(new URL(src).hostname);
    } catch {
      ok = false;
    }
    if (!ok) node.parentNode?.removeChild(node);
  }
});

DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A") {
    // Open external links safely.
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noopener noreferrer nofollow");
  }
  if (node.tagName === "IMG") {
    node.setAttribute("loading", "lazy");
  }
});

export function sanitizeEssayHtml(dirty: string): string {
  return DOMPurify.sanitize(dirty, {
    ALLOWED_TAGS,
    ALLOWED_ATTR,
    ALLOWED_URI_REGEXP: /^(?:https?:|mailto:|\/|#)/i,
    ADD_ATTR: ["target"],
  });
}

/** Plain-text projection of essay HTML, for excerpts and moderation input. */
export function htmlToText(html: string): string {
  const stripped = DOMPurify.sanitize(html, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] });
  return stripped.replace(/\s+/g, " ").trim();
}

export function makeExcerpt(html: string, max = 280): string {
  const text = htmlToText(html);
  if (text.length <= max) return text;
  return text.slice(0, max).replace(/\s+\S*$/, "") + "…";
}
