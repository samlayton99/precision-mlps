import type { Metadata } from "next";
import { siteConfig } from "@/config/site";

export const metadata: Metadata = { title: "Community Guidelines" };

export default function GuidelinesPage() {
  return (
    <div className="mx-auto max-w-2xl">
      <h1 className="font-serif text-4xl font-semibold">Community Guidelines</h1>
      <p className="mt-4 text-lg text-muted">
        {siteConfig.name} is a place to read and write thoughtful, faithful essays — to reason
        together in a way that helps us come closer to Jesus Christ and build Zion.
      </p>

      <div className="essay mt-8">
        <h2>Our purpose</h2>
        <p>
          We believe the life of the mind and the life of discipleship belong together. This is a
          space for careful, honest, intellectually serious writing offered in good faith and in
          the spirit of the gospel of Jesus Christ. Hard questions are welcome here; the aim is
          always to seek light, build one another up, and draw nearer to the Savior.
        </p>

        <h2>What we expect</h2>
        <ul>
          <li><strong>Charity first.</strong> Write and comment as disciples of Christ — with kindness, patience, and respect, even in disagreement.</li>
          <li><strong>Intellectual honesty.</strong> Argue in good faith, represent others fairly, cite sources, and be honest about uncertainty.</li>
          <li><strong>Faithfulness.</strong> Content should be consistent with and supportive of the doctrine of The Church of Jesus Christ of Latter-day Saints and its leaders — not in opposition to them.</li>
          <li><strong>Reverence.</strong> Treat sacred things with reverence and discretion.</li>
          <li><strong>Build Zion.</strong> Aim to edify. Ask whether your words help others come closer to Christ.</li>
        </ul>

        <h2>What isn&rsquo;t allowed</h2>
        <ul>
          <li>Content that attacks or seeks to undermine the Church, its leaders, or its doctrine (&ldquo;anti&rdquo; content).</li>
          <li>Hateful, demeaning, or harassing speech toward any person or group.</li>
          <li>Profanity, obscenity, or sexually explicit or graphic material.</li>
          <li>Personal attacks, contention for its own sake, and needless polemics.</li>
          <li>Sharing others&rsquo; private information; spam and self-promotion.</li>
        </ul>

        <h2>Honest questions are welcome</h2>
        <p>
          Sincere questions asked in faith are not only allowed — they are the point. The line is
          between honest inquiry that seeks understanding and content whose purpose is to tear down.
          When in doubt, ask in a way that invites light.
        </p>

        <h2>How moderation works</h2>
        <p>
          A daily review flags possible violations for human administrators, and members can report
          content for review. Administrators may warn, hide, or remove content, and may restrict
          accounts that repeatedly violate these guidelines. Comments and chat are text only.
          Leadership may refine these guidelines over time.
        </p>
      </div>
    </div>
  );
}
