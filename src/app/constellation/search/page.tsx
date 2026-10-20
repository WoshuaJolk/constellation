import type { Metadata } from "next";
import { FrameSearch } from "./search";

export const metadata: Metadata = {
  title: "Frame finder · const ellation;",
  description: "Search public astronomical image archives by sky field, dates, or candidate Solar System orbits.",
};
export default function Page() {
  return <FrameSearch />;
}
