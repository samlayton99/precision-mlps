/**
 * Database types.
 *
 * This mirrors the SQL migrations. You can regenerate it exactly from your live
 * project with:
 *   npx supabase gen types typescript --project-id <ref> --schema public > src/lib/types.ts
 * It is hand-authored here so the app type-checks before you connect Supabase.
 */

type Timestamp = string;
type UUID = string;
type Json = string | number | boolean | null | { [k: string]: Json } | Json[];

export type PostStatus = "published" | "hidden" | "draft";
export type FlagStatus = "open" | "dismissed" | "actioned";
export type NotificationType = "mention" | "comment" | "admin";

export interface Database {
  public: {
    Tables: {
      profiles: {
        Row: {
          id: UUID;
          handle: string;
          display_name: string;
          avatar_url: string | null;
          bio: string | null;
          is_banned: boolean;
          banned_reason: string | null;
          banned_at: Timestamp | null;
          banned_by: UUID | null;
          created_at: Timestamp;
          updated_at: Timestamp;
        };
        Insert: {
          id: UUID;
          handle: string;
          display_name: string;
          avatar_url?: string | null;
          bio?: string | null;
        };
        Update: {
          display_name?: string;
          avatar_url?: string | null;
          bio?: string | null;
          handle?: string;
        };
        Relationships: [];
      };
      admins: {
        Row: { user_id: UUID; granted_by: UUID | null; granted_at: Timestamp };
        Insert: { user_id: UUID; granted_by?: UUID | null };
        Update: { granted_by?: UUID | null };
        Relationships: [];
      };
      boards: {
        Row: {
          id: UUID;
          slug: string;
          name: string;
          description: string | null;
          created_by: UUID | null;
          is_archived: boolean;
          sort_order: number;
          created_at: Timestamp;
        };
        Insert: {
          slug: string;
          name: string;
          description?: string | null;
          sort_order?: number;
        };
        Update: {
          name?: string;
          description?: string | null;
          is_archived?: boolean;
          sort_order?: number;
        };
        Relationships: [];
      };
      posts: {
        Row: {
          id: UUID;
          board_id: UUID;
          author_id: UUID;
          title: string;
          subtitle: string | null;
          content_html: string;
          excerpt: string | null;
          cover_image_url: string | null;
          status: PostStatus;
          hidden_reason: string | null;
          hidden_by: UUID | null;
          hidden_at: Timestamp | null;
          like_count: number;
          comment_count: number;
          created_at: Timestamp;
          updated_at: Timestamp;
          published_at: Timestamp | null;
        };
        Insert: {
          board_id: UUID;
          author_id: UUID;
          title: string;
          subtitle?: string | null;
          content_html: string;
          excerpt?: string | null;
          cover_image_url?: string | null;
          status?: PostStatus;
        };
        Update: {
          board_id?: UUID;
          title?: string;
          subtitle?: string | null;
          content_html?: string;
          excerpt?: string | null;
          cover_image_url?: string | null;
          status?: PostStatus;
        };
        Relationships: [];
      };
      comments: {
        Row: {
          id: UUID;
          post_id: UUID;
          author_id: UUID;
          body: string;
          created_at: Timestamp;
          updated_at: Timestamp;
        };
        Insert: { post_id: UUID; author_id: UUID; body: string };
        Update: { body?: string };
        Relationships: [];
      };
      post_likes: {
        Row: { post_id: UUID; user_id: UUID; created_at: Timestamp };
        Insert: { post_id: UUID; user_id: UUID };
        Update: never;
        Relationships: [];
      };
      mentions: {
        Row: {
          id: UUID;
          source_type: "post" | "comment";
          source_id: UUID;
          mentioned_user_id: UUID;
          author_id: UUID;
          created_at: Timestamp;
        };
        Insert: {
          source_type: "post" | "comment";
          source_id: UUID;
          mentioned_user_id: UUID;
          author_id: UUID;
        };
        Update: never;
        Relationships: [];
      };
      board_chat_messages: {
        Row: {
          id: UUID;
          board_id: UUID;
          author_id: UUID;
          body: string;
          created_at: Timestamp;
        };
        Insert: { board_id: UUID; author_id: UUID; body: string };
        Update: never;
        Relationships: [];
      };
      notifications: {
        Row: {
          id: UUID;
          user_id: UUID;
          type: NotificationType;
          actor_id: UUID | null;
          post_id: UUID | null;
          comment_id: UUID | null;
          body: string | null;
          is_read: boolean;
          created_at: Timestamp;
        };
        Insert: {
          user_id: UUID;
          type: NotificationType;
          actor_id?: UUID | null;
          post_id?: UUID | null;
          comment_id?: UUID | null;
          body?: string | null;
        };
        Update: { is_read?: boolean };
        Relationships: [];
      };
      moderation_flags: {
        Row: {
          id: UUID;
          content_type: "post" | "comment" | "chat" | "profile";
          content_id: UUID;
          board_id: UUID | null;
          category: string | null;
          severity: "low" | "medium" | "high" | null;
          reason: string;
          flagged_by: "bot" | "user";
          reporter_id: UUID | null;
          status: FlagStatus;
          resolved_by: UUID | null;
          resolved_at: Timestamp | null;
          created_at: Timestamp;
        };
        Insert: {
          content_type: "post" | "comment" | "chat" | "profile";
          content_id: UUID;
          board_id?: UUID | null;
          category?: string | null;
          severity?: "low" | "medium" | "high" | null;
          reason: string;
          flagged_by?: "bot" | "user";
          reporter_id?: UUID | null;
          status?: FlagStatus;
        };
        Update: { status?: FlagStatus };
        Relationships: [];
      };
      admin_removal_requests: {
        Row: {
          id: UUID;
          target_admin_id: UUID;
          requested_by: UUID;
          reason: string | null;
          created_at: Timestamp;
        };
        Insert: { target_admin_id: UUID; requested_by: UUID; reason?: string | null };
        Update: never;
        Relationships: [];
      };
      admin_invites: {
        Row: { email: string; invited_by: UUID | null; created_at: Timestamp };
        Insert: { email: string; invited_by?: UUID | null };
        Update: never;
        Relationships: [];
      };
      audit_log: {
        Row: {
          id: number;
          actor_id: UUID | null;
          action: string;
          target_type: string | null;
          target_id: string | null;
          meta: Json;
          created_at: Timestamp;
        };
        Insert: never;
        Update: never;
        Relationships: [];
      };
      moderation_runs: {
        Row: {
          id: UUID;
          started_at: Timestamp;
          finished_at: Timestamp | null;
          items_scanned: number;
          flags_created: number;
          model: string | null;
          error: string | null;
        };
        Insert: {
          items_scanned?: number;
          flags_created?: number;
          model?: string | null;
          error?: string | null;
        };
        Update: {
          finished_at?: Timestamp | null;
          items_scanned?: number;
          flags_created?: number;
          error?: string | null;
        };
        Relationships: [];
      };
    };
    Views: Record<string, never>;
    Functions: {
      is_admin: { Args: { uid: UUID }; Returns: boolean };
      is_banned: { Args: { uid: UUID }; Returns: boolean };
      required_removal_votes: { Args: Record<string, never>; Returns: number };
      grant_admin: { Args: { p_target: UUID }; Returns: undefined };
      invite_admin_by_email: { Args: { p_email: string }; Returns: Json };
      ensure_admin_from_invite: { Args: Record<string, never>; Returns: boolean };
      self_revoke_admin: { Args: Record<string, never>; Returns: undefined };
      request_admin_removal: { Args: { p_target: UUID; p_reason?: string }; Returns: Json };
      cancel_admin_removal: { Args: { p_target: UUID }; Returns: undefined };
      set_user_ban: { Args: { p_target: UUID; p_banned: boolean; p_reason?: string }; Returns: undefined };
      admin_delete_post: { Args: { p_post: UUID; p_reason?: string }; Returns: undefined };
      admin_set_post_hidden: { Args: { p_post: UUID; p_hidden: boolean; p_reason?: string }; Returns: undefined };
      admin_delete_comment: { Args: { p_comment: UUID; p_reason?: string }; Returns: undefined };
      admin_delete_chat_message: { Args: { p_msg: UUID; p_reason?: string }; Returns: undefined };
      create_board: { Args: { p_slug: string; p_name: string; p_description?: string }; Returns: UUID };
      delete_board: { Args: { p_board: UUID }; Returns: undefined };
      resolve_moderation_flag: { Args: { p_flag: UUID; p_status: string }; Returns: undefined };
    };
    Enums: Record<string, never>;
    CompositeTypes: Record<string, never>;
  };
}

// Convenience row aliases used across the app.
export type Profile = Database["public"]["Tables"]["profiles"]["Row"];
export type Board = Database["public"]["Tables"]["boards"]["Row"];
export type Post = Database["public"]["Tables"]["posts"]["Row"];
export type Comment = Database["public"]["Tables"]["comments"]["Row"];
export type ChatMessage = Database["public"]["Tables"]["board_chat_messages"]["Row"];
export type Notification = Database["public"]["Tables"]["notifications"]["Row"];
export type ModerationFlag = Database["public"]["Tables"]["moderation_flags"]["Row"];

// Common joined shapes.
export type PostWithAuthor = Post & { author: Pick<Profile, "handle" | "display_name" | "avatar_url">; board?: Pick<Board, "slug" | "name"> };
export type CommentWithAuthor = Comment & { author: Pick<Profile, "handle" | "display_name" | "avatar_url"> };
