"use client";

import { useCallback, useRef } from "react";
import { useEditor, EditorContent, type Editor as TiptapEditor } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Underline from "@tiptap/extension-underline";
import LinkExt from "@tiptap/extension-link";
import Image from "@tiptap/extension-image";
import Youtube from "@tiptap/extension-youtube";
import Placeholder from "@tiptap/extension-placeholder";
import { createClient } from "@/lib/supabase/client";
import {
  Bold, Italic, Underline as U, Strikethrough, Heading2, Heading3,
  Quote, List, ListOrdered, Link2, ImageIcon, Youtube as YtIcon, Minus, Undo, Redo,
} from "lucide-react";
import { cn } from "@/lib/utils";

const STORAGE_BUCKET = "essay-media";

export function Editor({
  initialHtml,
  onChange,
}: {
  initialHtml?: string;
  onChange: (html: string) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);

  const editor = useEditor({
    immediatelyRender: false, // avoid Next SSR hydration mismatch
    extensions: [
      StarterKit.configure({ heading: { levels: [2, 3] } }),
      Underline,
      LinkExt.configure({ openOnClick: false, autolink: true, HTMLAttributes: { rel: "noopener noreferrer nofollow" } }),
      Image.configure({ HTMLAttributes: { class: "rounded-lg" } }),
      Youtube.configure({ width: 640, height: 360, nocookie: true }),
      Placeholder.configure({ placeholder: "Write your essay… (select text to format)" }),
    ],
    content: initialHtml ?? "",
    editorProps: {
      attributes: { class: "essay min-h-[24rem] focus:outline-none" },
    },
    onUpdate: ({ editor }) => onChange(editor.getHTML()),
  });

  const uploadImage = useCallback(
    async (file: File) => {
      if (!editor) return;
      if (file.size > 8 * 1024 * 1024) {
        alert("Images must be under 8 MB.");
        return;
      }
      const supabase = createClient();
      const ext = file.name.split(".").pop() ?? "png";
      const path = `posts/${crypto.randomUUID()}.${ext}`;
      const { error } = await supabase.storage.from(STORAGE_BUCKET).upload(path, file, {
        cacheControl: "3600",
        upsert: false,
        contentType: file.type,
      });
      if (error) {
        alert(`Image upload failed: ${error.message}. You can paste an image URL instead.`);
        return;
      }
      const { data } = supabase.storage.from(STORAGE_BUCKET).getPublicUrl(path);
      editor.chain().focus().setImage({ src: data.publicUrl }).run();
    },
    [editor],
  );

  if (!editor) return <div className="min-h-[24rem] animate-pulse rounded-lg bg-surface" />;

  return (
    <div className="rounded-xl border border-line bg-surface">
      <Toolbar editor={editor} onPickImage={() => fileRef.current?.click()} />
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) uploadImage(f);
          e.target.value = "";
        }}
      />
      <div className="px-5 py-4">
        <EditorContent editor={editor} />
      </div>
    </div>
  );
}

function Toolbar({ editor, onPickImage }: { editor: TiptapEditor; onPickImage: () => void }) {
  const setLink = () => {
    const prev = editor.getAttributes("link").href as string | undefined;
    const url = window.prompt("Link URL", prev ?? "https://");
    if (url === null) return;
    if (url === "") {
      editor.chain().focus().extendMarkRange("link").unsetLink().run();
      return;
    }
    editor.chain().focus().extendMarkRange("link").setLink({ href: url }).run();
  };

  const addYoutube = () => {
    const url = window.prompt("YouTube URL");
    if (url) editor.commands.setYoutubeVideo({ src: url });
  };

  return (
    <div className="flex flex-wrap items-center gap-0.5 border-b border-line px-2 py-1.5">
      <Btn on={editor.isActive("bold")} act={() => editor.chain().focus().toggleBold().run()} label="Bold"><Bold size={16} /></Btn>
      <Btn on={editor.isActive("italic")} act={() => editor.chain().focus().toggleItalic().run()} label="Italic"><Italic size={16} /></Btn>
      <Btn on={editor.isActive("underline")} act={() => editor.chain().focus().toggleUnderline().run()} label="Underline"><U size={16} /></Btn>
      <Btn on={editor.isActive("strike")} act={() => editor.chain().focus().toggleStrike().run()} label="Strikethrough"><Strikethrough size={16} /></Btn>
      <Divider />
      <Btn on={editor.isActive("heading", { level: 2 })} act={() => editor.chain().focus().toggleHeading({ level: 2 }).run()} label="Heading"><Heading2 size={16} /></Btn>
      <Btn on={editor.isActive("heading", { level: 3 })} act={() => editor.chain().focus().toggleHeading({ level: 3 }).run()} label="Subheading"><Heading3 size={16} /></Btn>
      <Btn on={editor.isActive("blockquote")} act={() => editor.chain().focus().toggleBlockquote().run()} label="Quote"><Quote size={16} /></Btn>
      <Btn on={editor.isActive("bulletList")} act={() => editor.chain().focus().toggleBulletList().run()} label="Bullet list"><List size={16} /></Btn>
      <Btn on={editor.isActive("orderedList")} act={() => editor.chain().focus().toggleOrderedList().run()} label="Numbered list"><ListOrdered size={16} /></Btn>
      <Divider />
      <Btn on={editor.isActive("link")} act={setLink} label="Link"><Link2 size={16} /></Btn>
      <Btn on={false} act={onPickImage} label="Image"><ImageIcon size={16} /></Btn>
      <Btn on={false} act={addYoutube} label="YouTube"><YtIcon size={16} /></Btn>
      <Btn on={false} act={() => editor.chain().focus().setHorizontalRule().run()} label="Divider"><Minus size={16} /></Btn>
      <Divider />
      <Btn on={false} act={() => editor.chain().focus().undo().run()} label="Undo"><Undo size={16} /></Btn>
      <Btn on={false} act={() => editor.chain().focus().redo().run()} label="Redo"><Redo size={16} /></Btn>
    </div>
  );
}

function Btn({ on, act, label, children }: { on: boolean; act: () => void; label: string; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={act}
      title={label}
      aria-label={label}
      aria-pressed={on}
      className={cn("rounded p-1.5 hover:bg-brand-soft", on && "bg-brand-soft text-brand")}
    >
      {children}
    </button>
  );
}

function Divider() {
  return <span className="mx-1 h-5 w-px bg-line" aria-hidden />;
}
