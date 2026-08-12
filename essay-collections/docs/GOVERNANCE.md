# Admin Governance

Essay Collections is stewarded by a small team of admins. Governance is deliberately simple, transparent, and hard to abuse: a hard cap on the number of admins, an append-only audit log, and — crucially — **no admin can unilaterally remove another admin**. Removing a peer takes a vote.

Everything here is enforced in the database, in `supabase/migrations/0003_functions.sql` (RPCs) and `0001_init.sql` (the cap trigger). Application code cannot bypass it; the RPCs re-check authority themselves.

---

## The 20-admin cap

There can be at most **20 admins**, enforced by the `enforce_admin_cap()` trigger on the `admins` table — any insert that would make the 21st admin is rejected at the database level. The invite/grant RPCs also check the cap before acting, so a would-be grant fails cleanly with a clear error rather than a raw trigger exception.

A small, fixed roster keeps accountability high and coordination cheap. Twenty is enough for a thriving community and small enough that everyone knows who holds the keys.

## What admins can do

Every admin power runs through a `SECURITY DEFINER` RPC that verifies the caller is an admin and writes an `audit_log` entry:

| Power | RPC |
| --- | --- |
| Delete a post | `admin_delete_post` |
| Hide / unhide ("pause") a post | `admin_set_post_hidden` |
| Delete a comment | `admin_delete_comment` |
| Delete a chat message | `admin_delete_chat_message` |
| Create a board | `create_board` |
| Delete a board | `delete_board` (cascades to its posts/comments/chat) |
| Ban / unban a user | `set_user_ban` |
| Resolve a moderation flag | `resolve_moderation_flag` |
| Grant admin to a user | `grant_admin` |
| Invite admin by email | `invite_admin_by_email` |
| Revoke your **own** admin | `self_revoke_admin` |
| Request removal of another admin | `request_admin_removal` |
| Withdraw your removal request | `cancel_admin_removal` |

## Adding an admin

Any admin may add another, two ways (both subject to the cap):

- **Grant to an existing user** — `grant_admin(user_id)`. Idempotent; refuses to grant to a **banned** user.
- **Invite by email** — `invite_admin_by_email(email)`. If that email already belongs to a signed-up user, admin is granted immediately. Otherwise a pending row is stored in `admin_invites`, and the grant is **applied automatically on that person's next sign-in** (via `ensure_admin_from_invite()`, called by the app). If the cap is full when they sign in, the invite is kept and applies later when a slot opens.

The very first admins are seeded from the `BOOTSTRAP_ADMIN_EMAILS` env allowlist on first sign-in — see [DEPLOY.md](DEPLOY.md) step (h).

## Removing an admin

There are two distinct paths, and only one requires a vote:

### Self-revoke (no vote)

You may always step down yourself with `self_revoke_admin()`. It removes you from `admins` and clears any removal requests involving you.

### Removing someone else (requires a vote)

No single admin can remove a peer. Instead, an admin files a removal request against a target with `request_admin_removal(target, reason)`. Requests are deduplicated per (target, requester). **When the number of *distinct other admins* who have filed a request against the same target reaches the threshold, the target is removed automatically and atomically** — the removal happens inside the same RPC call that crosses the line, and an `admin.removed_by_vote` audit entry is written. A requester may withdraw with `cancel_admin_removal(target)` before the threshold is reached.

The target's own vote never counts — the tally only includes admins where `user_id <> target`.

### The threshold formula

```sql
required_removal_votes() = LEAST(10, floor(current_admin_count / 2) + 1)
```

In words: **the smaller of 10 and a strict majority of the current admins.** A strict majority (`floor(N/2)+1`) prevents a tie or a minority faction from removing someone; the cap of 10 keeps removal achievable even on a full 20-person roster (where a bare majority would be 11).

`current_admin_count` is the total number of admins *including* the target; the votes themselves must come from the *other* admins.

### Worked examples

`N` = current number of admins (including the target). "Other admins" = `N − 1` (the largest possible tally). "Votes needed" = `required_removal_votes()`.

| Admins (`N`) | `floor(N/2)+1` | Votes needed = `LEAST(10, …)` | Other admins available | Note |
| ---: | ---: | ---: | ---: | --- |
| 2  | 2  | 2  | 1  | Impossible by vote — only 1 other admin. Use self-revoke. |
| 3  | 2  | 2  | 2  | Both other admins must agree (unanimous). |
| 5  | 3  | 3  | 4  | 3 of the other 4. |
| 6  | 4  | **4**  | 5  | 4 of the other 5. |
| 10 | 6  | 6  | 9  | 6 of the other 9. |
| 12 | 7  | **7**  | 11 | 7 of the other 11. |
| 19 | 10 | 10 | 18 | Majority would be 10 — cap not yet binding. |
| 20 | 11 | **10** | 19 | The `LEAST(10, …)` cap binds: 10, not 11. |

So on a full 20-admin roster, **10 other admins** filing removal requests against the same person removes them.

### Making removal stricter

The threshold is intentionally one line. To require **"10 *and* a majority"** (i.e. the *larger* of the two — harder to remove someone), change `LEAST` to `GREATEST` in `required_removal_votes()` in `0003_functions.sql`:

```sql
-- stricter: need BOTH 10 votes AND a majority
select greatest(10, ((select count(*) from public.admins) / 2)::int + 1);
```

## Admins cannot be banned while they are admins

`set_user_ban()` **refuses** to ban a user who is currently an admin — it raises "remove admin privileges before banning an admin." This prevents an end-run around the removal vote (you can't silence a peer by banning them). The correct order is: remove admin (self-revoke or vote), then ban if warranted.

## The audit log

Every privileged action writes an append-only row to `audit_log` (actor, action, target, JSON metadata, timestamp) via the internal `_audit()` helper. It is **admin-readable** and no RPC updates or deletes it — it is the tamper-evident record of who did what. Governance events recorded include `admin.grant`, `admin.invite`, `admin.invite.accept`, `admin.self_revoke`, `admin.removal_request`, and `admin.removed_by_vote` (the last two carry the vote tally and the required threshold in `meta`).

## Where to change the thresholds in code

| To change | Edit |
| --- | --- |
| The removal-vote threshold | `required_removal_votes()` in `0003_functions.sql` (`LEAST` → `GREATEST` for stricter) |
| The 20-admin hard cap | The `>= 20` check in the `enforce_admin_cap()` trigger (`0001_init.sql`) **and** the `>= 20` checks in `invite_admin_by_email` / `ensure_admin_from_invite` (`0003_functions.sql`) — change all of them together |

## Healthy admin culture

Rules constrain the worst case; culture sets the norm. A few principles that keep a small stewardship healthy:

- **Prefer the gentlest effective action.** Warn or hide before you delete; delete before you ban. Removal and bans are last resorts.
- **Write real reasons.** Every RPC accepts a reason and records it. Future admins (and the affected member) deserve to understand a decision.
- **Assume good faith.** Most flagged content is a misunderstanding, not malice. See the [Community Guidelines](COMMUNITY_GUIDELINES.md) — hard, sincere questions asked in faith are welcome.
- **Removal is for stewardship, not for winning arguments.** The vote exists to remove someone who is harming the community or abusing the tools — never to settle a disagreement about ideas.
- **Keep the roster lean and trusted.** Empty the bootstrap allowlist after launch, and grant admin sparingly.

Admins and leadership may refine these governance practices over time; the enforced rules above are the floor, not the ceiling.
