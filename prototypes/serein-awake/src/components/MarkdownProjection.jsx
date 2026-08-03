import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function markdownSource(content) {
  if (Array.isArray(content)) return content.join("\n\n");
  return typeof content === "string" ? content : "";
}

export function MarkdownProjection({ content, className = "" }) {
  const markdown = markdownSource(content);

  return (
    <div className={`markdown-projection${className ? ` ${className}` : ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          a: ({ node: _node, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer" />
          ),
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
