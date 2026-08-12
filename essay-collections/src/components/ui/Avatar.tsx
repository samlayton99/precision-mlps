/* eslint-disable @next/next/no-img-element */
import { cn } from "@/lib/utils";

/**
 * Small avatar. Uses a plain <img> (not next/image) so arbitrary Google /
 * Supabase avatar hosts render without configuring every domain. Falls back to
 * an initial when there is no picture.
 */
export function Avatar({
  src,
  name,
  size = 36,
  className,
}: {
  src?: string | null;
  name: string;
  size?: number;
  className?: string;
}) {
  const initial = name?.trim()?.[0]?.toUpperCase() ?? "?";
  return src ? (
    <img
      src={src}
      alt={name}
      width={size}
      height={size}
      referrerPolicy="no-referrer"
      className={cn("rounded-full object-cover", className)}
      style={{ width: size, height: size }}
    />
  ) : (
    <span
      aria-hidden
      className={cn(
        "inline-flex items-center justify-center rounded-full bg-brand-soft font-medium text-brand",
        className,
      )}
      style={{ width: size, height: size, fontSize: size * 0.45 }}
    >
      {initial}
    </span>
  );
}
