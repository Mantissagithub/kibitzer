import {
  isValidElement,
  type ComponentPropsWithoutRef,
  type ComponentType,
  type ReactNode,
} from "react";
import type { MDXComponents } from "mdx/types";
import { ZoomableFigure } from "./ZoomableFigure";

type LogbookIndex = {
  imageDimensions: Record<string, { width: number; height: number }>;
};

function textFromChildren(children: ReactNode): string {
  if (typeof children === "string" || typeof children === "number") return String(children);
  if (Array.isArray(children)) return children.map(textFromChildren).join("");
  if (isValidElement<{ children?: ReactNode }>(children)) {
    return textFromChildren(children.props.children);
  }
  return "";
}

export function slugifyHeading(text: string) {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}

function headingId(children: ReactNode) {
  return slugifyHeading(textFromChildren(children));
}

function resolveHref(href: string) {
  if (!href || href.startsWith("#") || /^(https?:)?\/\//.test(href)) return href;
  if (href === "LOGBOOK.md") return "/logbook";
  return `https://github.com/Mantissagithub/kibitzer/blob/main/${href.replace(/^\.\//, "")}`;
}

export function createLogbookComponents(index: LogbookIndex): MDXComponents {
  return {
    h1: ({ children, ...props }: ComponentPropsWithoutRef<"h1">) => (
      <h1 id={headingId(children)} {...props}>{children}</h1>
    ),
    h2: ({ children, ...props }: ComponentPropsWithoutRef<"h2">) => (
      <h2 id={headingId(children)} {...props}>{children}</h2>
    ),
    h3: ({ children, ...props }: ComponentPropsWithoutRef<"h3">) => (
      <h3 id={headingId(children)} data-logbook-decision={textFromChildren(children).match(/^(D\d+)\./)?.[1]} {...props}>
        {children}
      </h3>
    ),
    h4: ({ children, ...props }: ComponentPropsWithoutRef<"h4">) => (
      <h4 id={headingId(children)} {...props}>{children}</h4>
    ),
    a: ({ href = "", ...props }: ComponentPropsWithoutRef<"a">) => {
      const resolved = resolveHref(href);
      const external = /^(https?:)?\/\//.test(resolved);
      return (
        <a
          href={resolved}
          {...(external ? { target: "_blank", rel: "noreferrer" } : {})}
          {...props}
        />
      );
    },
    img: ({ src, alt }: ComponentPropsWithoutRef<"img">) => {
      const normalized = typeof src === "string" ? src : "";
      const dimensions = index.imageDimensions[normalized];
      return (
        <ZoomableFigure
          src={normalized}
          alt={alt ?? ""}
          width={dimensions?.width}
          height={dimensions?.height}
        />
      );
    },
    table: (props: ComponentPropsWithoutRef<"table">) => (
      <div className="table-wrap"><table {...props} /></div>
    ),
    pre: ({ children, ...props }: ComponentPropsWithoutRef<"pre">) => (
      <pre {...props}>{children}</pre>
    ),
  };
}

export function LogbookDocument({
  content: Content,
  index,
}: {
  content: ComponentType<{ components?: MDXComponents }>;
  index: LogbookIndex;
}) {
  return (
    <article className="reading-article min-w-0">
      <Content components={createLogbookComponents(index)} />
    </article>
  );
}
