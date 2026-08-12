"use client";

import { useState, useTransition } from "react";
import { Heart } from "lucide-react";
import { toggleLike } from "@/app/actions/likes";
import { cn } from "@/lib/utils";

export function LikeButton({
  postId,
  initialLiked,
  initialCount,
  canInteract,
}: {
  postId: string;
  initialLiked: boolean;
  initialCount: number;
  canInteract: boolean;
}) {
  const [liked, setLiked] = useState(initialLiked);
  const [count, setCount] = useState(initialCount);
  const [pending, startTransition] = useTransition();

  function onClick() {
    if (!canInteract) {
      window.location.href = "/login";
      return;
    }
    // optimistic
    const nextLiked = !liked;
    setLiked(nextLiked);
    setCount((c) => c + (nextLiked ? 1 : -1));
    startTransition(async () => {
      const res = await toggleLike(postId);
      if (!res.ok) {
        setLiked(liked);
        setCount(initialCount);
      } else {
        setLiked(res.liked);
      }
    });
  }

  return (
    <button
      onClick={onClick}
      disabled={pending}
      aria-pressed={liked}
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-4 py-1.5 text-sm transition",
        liked ? "border-brand bg-brand-soft text-brand" : "border-line hover:bg-brand-soft",
      )}
    >
      <Heart size={16} fill={liked ? "currentColor" : "none"} />
      <span>{count}</span>
    </button>
  );
}
