import { z } from "zod";

/** Zod schemas — the single validation contract for every mutation. */

export const boardSlug = z
  .string()
  .trim()
  .toLowerCase()
  .regex(/^[a-z0-9]([a-z0-9-]{0,48}[a-z0-9])?$/, "Use lowercase letters, numbers, and hyphens.");

export const postSchema = z.object({
  boardId: z.string().uuid(),
  title: z.string().trim().min(1, "A title is required.").max(200),
  subtitle: z.string().trim().max(300).optional().or(z.literal("")),
  // Raw HTML from the editor; sanitized server-side before it is stored.
  contentHtml: z.string().min(1, "The essay is empty.").max(500_000),
  coverImageUrl: z.string().url().max(2000).optional().or(z.literal("")),
  status: z.enum(["published", "draft"]).default("published"),
});

export const commentSchema = z.object({
  postId: z.string().uuid(),
  body: z.string().trim().min(1, "Say something first.").max(10_000),
});

export const chatSchema = z.object({
  boardId: z.string().uuid(),
  body: z.string().trim().min(1).max(2000),
});

export const boardSchema = z.object({
  slug: boardSlug,
  name: z.string().trim().min(1).max(80),
  description: z.string().trim().max(500).optional().or(z.literal("")),
});

export const profileSchema = z.object({
  displayName: z.string().trim().min(1, "A display name is required.").max(60),
  handle: z
    .string()
    .trim()
    .toLowerCase()
    .regex(/^[a-z0-9_]{3,24}$/, "3–24 characters: lowercase letters, numbers, underscores."),
  bio: z.string().trim().max(500).optional().or(z.literal("")),
});

export const reportSchema = z.object({
  contentType: z.enum(["post", "comment", "chat", "profile"]),
  contentId: z.string().uuid(),
  reason: z.string().trim().min(3, "Tell us what's wrong.").max(1000),
});

export const banSchema = z.object({
  userId: z.string().uuid(),
  banned: z.boolean(),
  reason: z.string().trim().max(500).optional().or(z.literal("")),
});

export type PostInput = z.infer<typeof postSchema>;
export type CommentInput = z.infer<typeof commentSchema>;
export type BoardInput = z.infer<typeof boardSchema>;
