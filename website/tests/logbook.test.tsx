import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import LogbookContent from "@/generated/logbook.mdx";

describe("generated logbook math", () => {
  it("renders the source equations through KaTeX", () => {
    const { container } = render(<LogbookContent />);

    expect(container.querySelectorAll(".katex-display")).toHaveLength(5);
    expect(container.querySelectorAll(".katex").length).toBeGreaterThan(5);
    expect(container.querySelector(".katex-mathml math")).not.toBeNull();
  });
});
