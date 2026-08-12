"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { createClient } from "@/lib/supabase/client";
import { sendChatMessage, deleteOwnChatMessage } from "@/app/actions/chat";
import { Avatar } from "@/components/ui/Avatar";
import { timeAgo } from "@/lib/utils";
import { Send, X } from "lucide-react";

interface ChatAuthor {
  handle: string;
  display_name: string;
  avatar_url: string | null;
}
export interface ChatItem {
  id: string;
  author_id: string;
  body: string;
  created_at: string;
  author: ChatAuthor | null;
}
interface Viewer {
  id: string;
  isAdmin: boolean;
}

export function BoardChat({
  boardId,
  initialMessages,
  viewer,
}: {
  boardId: string;
  initialMessages: ChatItem[];
  viewer: Viewer | null;
}) {
  const [messages, setMessages] = useState<ChatItem[]>(initialMessages);
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const authorCache = useRef(new Map<string, ChatAuthor>());

  // seed the author cache from the initial page load
  useEffect(() => {
    for (const m of initialMessages) if (m.author) authorCache.current.set(m.author_id, m.author);
  }, [initialMessages]);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      const el = scrollRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    });
  }, []);

  useEffect(scrollToBottom, [messages.length, scrollToBottom]);

  useEffect(() => {
    const supabase = createClient();
    const channel = supabase
      .channel(`board-chat:${boardId}`)
      .on(
        "postgres_changes",
        { event: "INSERT", schema: "public", table: "board_chat_messages", filter: `board_id=eq.${boardId}` },
        async (payload) => {
          const row = payload.new as Omit<ChatItem, "author">;
          let author = authorCache.current.get(row.author_id) ?? null;
          if (!author) {
            const { data } = await supabase
              .from("profiles")
              .select("handle,display_name,avatar_url")
              .eq("id", row.author_id)
              .single();
            author = data ?? null;
            if (author) authorCache.current.set(row.author_id, author);
          }
          setMessages((prev) =>
            prev.some((m) => m.id === row.id) ? prev : [...prev, { ...row, author }],
          );
        },
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [boardId]);

  async function onSend(e: React.FormEvent) {
    e.preventDefault();
    const body = text.trim();
    if (!body || sending) return;
    setSending(true);
    setError(null);
    const res = await sendChatMessage({ boardId, body });
    setSending(false);
    if (res.ok) {
      setText("");
    } else {
      setError(res.error);
    }
  }

  async function onDelete(id: string) {
    setMessages((prev) => prev.filter((m) => m.id !== id)); // optimistic
    await deleteOwnChatMessage(id);
  }

  return (
    <div className="flex h-[32rem] flex-col rounded-xl border border-line bg-surface">
      <div className="border-b border-line px-4 py-3">
        <h3 className="text-sm font-medium">Board chat</h3>
        <p className="text-xs text-muted">Live, text-only conversation for this board.</p>
      </div>

      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-4 py-3">
        {messages.length === 0 && (
          <p className="py-8 text-center text-sm text-muted">No messages yet. Say hello.</p>
        )}
        {messages.map((m) => (
          <div key={m.id} className="group flex gap-2.5">
            <Avatar src={m.author?.avatar_url} name={m.author?.display_name ?? "?"} size={26} />
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-2">
                <span className="text-sm font-medium">{m.author?.display_name ?? "Member"}</span>
                <span className="text-xs text-muted">{timeAgo(m.created_at)}</span>
              </div>
              <p className="whitespace-pre-wrap break-words text-sm">{m.body}</p>
            </div>
            {viewer && (viewer.id === m.author_id) && (
              <button
                onClick={() => onDelete(m.id)}
                className="opacity-0 transition group-hover:opacity-100"
                title="Delete"
              >
                <X size={14} className="text-muted hover:text-danger" />
              </button>
            )}
          </div>
        ))}
      </div>

      {viewer ? (
        <form onSubmit={onSend} className="border-t border-line p-3">
          {error && <p className="mb-2 text-xs text-danger">{error}</p>}
          <div className="flex items-center gap-2">
            <input
              value={text}
              onChange={(e) => setText(e.target.value)}
              maxLength={2000}
              placeholder="Message the board…"
              className="flex-1 rounded-full border border-line bg-canvas px-4 py-2 text-sm focus:border-brand focus:outline-none"
            />
            <button
              type="submit"
              disabled={sending || !text.trim()}
              className="rounded-full bg-brand p-2 text-white disabled:opacity-50"
              title="Send"
            >
              <Send size={16} />
            </button>
          </div>
        </form>
      ) : (
        <div className="border-t border-line p-3 text-center text-sm text-muted">
          <a href="/login" className="text-brand underline">Sign in</a> to join the chat.
        </div>
      )}
    </div>
  );
}
