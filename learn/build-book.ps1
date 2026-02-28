# build-book.ps1 — Assemble learn/chapters/*.md into learn/book.html
# Usage: pwsh learn/build-book.ps1   (from repo root)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$chapDir   = Join-Path $scriptDir 'chapters'
$outFile   = Join-Path $scriptDir 'book.html'

# ── Gather chapter files ─────────────────────────────────────────────
$mdFiles = Get-ChildItem -Path $chapDir -Filter '*.md' | Sort-Object Name
if ($mdFiles.Count -eq 0) {
    Write-Error "No .md files found in $chapDir"
    exit 1
}

Write-Host "Found $($mdFiles.Count) chapters"

# ── Markdown → HTML converter ────────────────────────────────────────

function Format-Inline {
    param([string]$text)
    $text = [regex]::Replace($text, '!\[([^\]]*)\]\(([^)]+)\)', '<img src="$2" alt="$1">')
    $text = [regex]::Replace($text, '\[([^\]]+)\]\(([^)]+)\)', '<a href="$2">$1</a>')
    $text = [regex]::Replace($text, '\*\*\*(.+?)\*\*\*', '<strong><em>$1</em></strong>')
    $text = [regex]::Replace($text, '\*\*(.+?)\*\*', '<strong>$1</strong>')
    $text = [regex]::Replace($text, '(?<![\\])\*(.+?)\*', '<em>$1</em>')
    $text = [regex]::Replace($text, '`([^`]+)`', '<code>$1</code>')
    return $text
}

function Render-Table {
    param($tableRows, $tableAligns)
    $thtml = '<div class="table-wrapper"><table>'
    for ($ti = 0; $ti -lt $tableRows.Count; $ti++) {
        $cells = $tableRows[$ti] -replace '^\|' -replace '\|$' -split '\|'
        $tag = if ($ti -eq 0) { 'th' } else { 'td' }
        $thtml += '<tr>'
        for ($ci = 0; $ci -lt $cells.Count; $ci++) {
            $align = ''
            if ($tableAligns.Count -gt $ci -and $tableAligns[$ci]) {
                $align = " style=`"text-align:$($tableAligns[$ci])`""
            }
            $cellText = Format-Inline $cells[$ci].Trim()
            $thtml += "<$tag$align>$cellText</$tag>"
        }
        $thtml += '</tr>'
    }
    $thtml += '</table></div>'
    return $thtml
}

function Convert-MarkdownToHtml {
    param([string]$md, [int]$chapterNum)

    $lines = $md -split "`n"
    $html  = [System.Collections.Generic.List[string]]::new()
    $i = 0

    $inCodeBlock  = $false
    $codeLang     = ''
    $codeLines    = [System.Collections.Generic.List[string]]::new()
    $inMathBlock  = $false
    $mathLines    = [System.Collections.Generic.List[string]]::new()
    $inUL         = $false
    $inOL         = $false
    $inBlockquote = $false
    $inTable      = $false
    $tableRows    = [System.Collections.Generic.List[string]]::new()
    $tableAligns  = @()

    while ($i -lt $lines.Count) {
        $line = $lines[$i].TrimEnd("`r")

        # ── Code fence ──────────────────────────
        if ($line -match '^```(.*)$') {
            if ($inCodeBlock) {
                $code = ($codeLines -join "`n")
                $escaped = [System.Web.HttpUtility]::HtmlEncode($code)
                if ($codeLang -eq 'mermaid') {
                    $html.Add("<div class=`"mermaid-wrapper`"><pre class=`"mermaid`">$escaped</pre></div>")
                } else {
                    $badge = if ($codeLang) { "<span class=`"lang-badge`">$codeLang</span>" } else { '' }
                    $html.Add("<div class=`"code-wrapper`">$badge<pre><code class=`"language-$codeLang`">$escaped</code></pre></div>")
                }
                $codeLines.Clear()
                $inCodeBlock = $false
            } else {
                if ($inUL)  { $html.Add('</ul>');  $inUL = $false }
                if ($inOL)  { $html.Add('</ol>');  $inOL = $false }
                if ($inBlockquote) { $html.Add('</blockquote>'); $inBlockquote = $false }
                if ($inTable) { $html.Add((Render-Table $tableRows $tableAligns)); $tableRows.Clear(); $tableAligns = @(); $inTable = $false }
                $codeLang = $Matches[1].Trim()
                $inCodeBlock = $true
            }
            $i++; continue
        }
        if ($inCodeBlock) {
            $codeLines.Add($line)
            $i++; continue
        }

        # ── Display math block $$...$$  ──────────
        if ($line -match '^\$\$' -and -not $inMathBlock) {
            if ($inUL)  { $html.Add('</ul>');  $inUL = $false }
            if ($inOL)  { $html.Add('</ol>');  $inOL = $false }
            if ($inBlockquote) { $html.Add('</blockquote>'); $inBlockquote = $false }
            if ($inTable) { $html.Add((Render-Table $tableRows $tableAligns)); $tableRows.Clear(); $tableAligns = @(); $inTable = $false }
            if ($line -match '^\$\$(.+)\$\$$') {
                $html.Add("<div class=`"math-block`">`$`$$($Matches[1])`$`$</div>")
                $i++; continue
            }
            $inMathBlock = $true
            $mathLines.Add($line)
            $i++; continue
        }
        if ($inMathBlock) {
            $mathLines.Add($line)
            if ($line -match '\$\$\s*$') {
                $mathContent = ($mathLines -join "`n")
                $html.Add("<div class=`"math-block`">$mathContent</div>")
                $mathLines.Clear()
                $inMathBlock = $false
            }
            $i++; continue
        }

        # ── Blank line ──────────────────────────
        if ($line -match '^\s*$') {
            if ($inUL)  { $html.Add('</ul>');  $inUL = $false }
            if ($inOL)  { $html.Add('</ol>');  $inOL = $false }
            if ($inBlockquote) { $html.Add('</blockquote>'); $inBlockquote = $false }
            if ($inTable) { $html.Add((Render-Table $tableRows $tableAligns)); $tableRows.Clear(); $tableAligns = @(); $inTable = $false }
            $i++; continue
        }

        # ── Headings ────────────────────────────
        if ($line -match '^(#{1,6})\s+(.+)$') {
            if ($inUL)  { $html.Add('</ul>');  $inUL = $false }
            if ($inOL)  { $html.Add('</ol>');  $inOL = $false }
            if ($inBlockquote) { $html.Add('</blockquote>'); $inBlockquote = $false }
            if ($inTable) { $html.Add((Render-Table $tableRows $tableAligns)); $tableRows.Clear(); $tableAligns = @(); $inTable = $false }
            $level = $Matches[1].Length
            $text  = Format-Inline $Matches[2]
            $slug  = ($Matches[2] -replace '[^a-zA-Z0-9 ]','' -replace '\s+','-').ToLower()
            if ($level -eq 1) {
                $html.Add("<h1 id=`"$slug`"><span class=`"chapter-watermark`">$chapterNum</span>$text</h1>")
            } else {
                $html.Add("<h$level id=`"$slug`">$text</h$level>")
            }
            $i++; continue
        }

        # ── Horizontal rule ─────────────────────
        if ($line -match '^-{3,}\s*$' -or $line -match '^\*{3,}\s*$') {
            if ($inUL)  { $html.Add('</ul>');  $inUL = $false }
            if ($inOL)  { $html.Add('</ol>');  $inOL = $false }
            if ($inBlockquote) { $html.Add('</blockquote>'); $inBlockquote = $false }
            if ($inTable) { $html.Add((Render-Table $tableRows $tableAligns)); $tableRows.Clear(); $tableAligns = @(); $inTable = $false }
            $html.Add('<hr>')
            $i++; continue
        }

        # ── Table row ───────────────────────────
        if ($line -match '^\|') {
            if ($inUL)  { $html.Add('</ul>');  $inUL = $false }
            if ($inOL)  { $html.Add('</ol>');  $inOL = $false }
            if ($inBlockquote) { $html.Add('</blockquote>'); $inBlockquote = $false }
            # check if separator row
            if ($line -match '^\|[\s\-:|]+\|$' -and $line -match '-') {
                $cols = $line -replace '^\|' -replace '\|$' -split '\|'
                $aligns = @()
                foreach ($col in $cols) {
                    $c = $col.Trim()
                    if ($c -match '^:-+:$')     { $aligns += 'center' }
                    elseif ($c -match '-+:$')   { $aligns += 'right' }
                    elseif ($c -match '^:-+$')  { $aligns += 'left' }
                    else                        { $aligns += '' }
                }
                $tableAligns = $aligns
                $inTable = $true
                $i++; continue
            }
            if (-not $inTable) {
                $inTable = $true
                $tableRows.Add($line)
                $i++; continue
            }
            $tableRows.Add($line)
            $i++; continue
        } elseif ($inTable) {
            $html.Add((Render-Table $tableRows $tableAligns))
            $tableRows.Clear()
            $tableAligns = @()
            $inTable = $false
        }

        # ── Blockquote ──────────────────────────
        if ($line -match '^>\s?(.*)$') {
            if ($inUL) { $html.Add('</ul>'); $inUL = $false }
            if ($inOL) { $html.Add('</ol>'); $inOL = $false }
            if ($inTable) { $html.Add((Render-Table $tableRows $tableAligns)); $tableRows.Clear(); $tableAligns = @(); $inTable = $false }
            if (-not $inBlockquote) {
                $html.Add('<blockquote>')
                $inBlockquote = $true
            }
            $content = Format-Inline $Matches[1]
            $html.Add("<p>$content</p>")
            $i++; continue
        } elseif ($inBlockquote) {
            $html.Add('</blockquote>')
            $inBlockquote = $false
        }

        # ── Unordered list ──────────────────────
        if ($line -match '^[-*+]\s+(.+)$') {
            if ($inBlockquote) { $html.Add('</blockquote>'); $inBlockquote = $false }
            if ($inOL) { $html.Add('</ol>'); $inOL = $false }
            if (-not $inUL) { $html.Add('<ul>'); $inUL = $true }
            $html.Add("<li>$(Format-Inline $Matches[1])</li>")
            $i++; continue
        }

        # ── Ordered list ────────────────────────
        if ($line -match '^\d+\.\s+(.+)$') {
            if ($inBlockquote) { $html.Add('</blockquote>'); $inBlockquote = $false }
            if ($inUL) { $html.Add('</ul>'); $inUL = $false }
            if (-not $inOL) { $html.Add('<ol>'); $inOL = $true }
            $html.Add("<li>$(Format-Inline $Matches[1])</li>")
            $i++; continue
        }

        # ── Exercise callout ────────────────────
        if ($line -match '^Exercise:\s*(.+)$') {
            if ($inUL)  { $html.Add('</ul>');  $inUL = $false }
            if ($inOL)  { $html.Add('</ol>');  $inOL = $false }
            if ($inBlockquote) { $html.Add('</blockquote>'); $inBlockquote = $false }
            if ($inTable) { $html.Add((Render-Table $tableRows $tableAligns)); $tableRows.Clear(); $tableAligns = @(); $inTable = $false }
            $exerciseText = Format-Inline $Matches[1]
            $html.Add("<div class=`"exercise`"><strong>Exercise:</strong> $exerciseText</div>")
            $i++; continue
        }

        # ── Regular paragraph ───────────────────
        if ($inUL) { $html.Add('</ul>'); $inUL = $false }
        if ($inOL) { $html.Add('</ol>'); $inOL = $false }
        $html.Add("<p>$(Format-Inline $line)</p>")
        $i++
    }

    # close any open elements
    if ($inUL) { $html.Add('</ul>') }
    if ($inOL) { $html.Add('</ol>') }
    if ($inBlockquote) { $html.Add('</blockquote>') }
    if ($inTable) { $html.Add((Render-Table $tableRows $tableAligns)) }

    return ($html -join "`n")
}

# ── Process chapters ─────────────────────────────────────────────────
Add-Type -AssemblyName System.Web

$tocEntries   = [System.Collections.Generic.List[string]]::new()
$sectionHtmls = [System.Collections.Generic.List[string]]::new()

foreach ($file in $mdFiles) {
    $md = Get-Content -Path $file.FullName -Raw -Encoding UTF8

    # Extract chapter number and title from first heading
    if ($md -match '# Chapter (\d+)\s*[—–-]\s*(.+)') {
        $chapNum   = [int]$Matches[1]
        $chapTitle = $Matches[2].Trim()
    } else {
        $chapNum   = [int]($file.BaseName -replace '^(\d+).*','$1')
        $chapTitle = $file.BaseName -replace '^\d+-','' -replace '-',' '
    }

    $chapId = "chapter-$chapNum"

    Write-Host "  Chapter $chapNum — $chapTitle"

    $bodyHtml = Convert-MarkdownToHtml -md $md -chapterNum $chapNum

    $tocEntries.Add("<a href=`"#$chapId`" class=`"toc-link`" data-chapter=`"$chapId`">$chapNum. $chapTitle</a>")
    $sectionHtmls.Add("<section id=`"$chapId`" class=`"chapter`">$bodyHtml</section>")
}

$tocHtml     = $tocEntries -join "`n"
$sectionsHtml = $sectionHtmls -join "`n"
$chapterCount = $mdFiles.Count

# ── HTML template (LITERAL here-string — do NOT use @"..."@) ────────
$template = @'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>nanochat — The Book</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<style>
/* ── Reset & Variables ───────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#faf9f7;--fg:#1a1a1a;--sidebar-bg:#f0eeeb;--sidebar-fg:#444;
  --accent:#2563eb;--accent-light:#dbeafe;
  --code-bg:#1e1e2e;--code-fg:#cdd6f4;
  --border:#e0ddd8;--card-bg:#fff;
  --math-bg:#f0f4ff;--mermaid-bg:#fff;
  --exercise-bg:#f0fdf4;--exercise-border:#22c55e;
  --blockquote-bg:#f8f7f5;--blockquote-border:#2563eb;
  --table-stripe:#f8f7f5;
  --progress:#2563eb;
  --cover-from:#1e3a5f;--cover-to:#2563eb;
  --serif: Georgia, Cambria, 'Times New Roman', serif;
  --sans: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  --mono: 'Cascadia Code', 'Fira Code', Consolas, 'Courier New', monospace;
  --line-height: 1.7;
}
@media(prefers-color-scheme:dark){
  :root{
    --bg:#181818;--fg:#e0e0e0;--sidebar-bg:#1e1e1e;--sidebar-fg:#b0b0b0;
    --accent:#60a5fa;--accent-light:#1e3a5f;
    --border:#333;--card-bg:#222;
    --math-bg:#1a2332;--mermaid-bg:#222;
    --exercise-bg:#0a1f0a;--exercise-border:#22c55e;
    --blockquote-bg:#222;--blockquote-border:#60a5fa;
    --table-stripe:#222;
    --cover-from:#0f1f33;--cover-to:#1e3a5f;
  }
}

/* ── Progress bar ────────────────────────────────── */
#progress-bar{
  position:fixed;top:0;left:0;height:3px;
  background:var(--progress);width:0;z-index:9999;
  transition:width .15s ease-out;
}

/* ── Layout ──────────────────────────────────────── */
body{
  font-family:var(--sans);font-size:17px;line-height:var(--line-height);
  color:var(--fg);background:var(--bg);
  display:grid;grid-template-columns:280px 1fr;min-height:100vh;
}
@media(max-width:900px){
  body{grid-template-columns:1fr}
  #sidebar{display:none}
}

/* ── Sidebar ─────────────────────────────────────── */
#sidebar{
  position:sticky;top:0;height:100vh;overflow-y:auto;
  background:var(--sidebar-bg);border-right:1px solid var(--border);
  padding:1.5rem 1rem;
}
#sidebar h2{
  font-family:var(--serif);font-size:1.1rem;margin-bottom:1rem;
  color:var(--sidebar-fg);letter-spacing:.02em;
}
.toc-link{
  display:block;padding:.4rem .6rem;margin:.15rem 0;
  border-radius:6px;text-decoration:none;font-size:.9rem;
  color:var(--sidebar-fg);transition:background .15s,color .15s;
}
.toc-link:hover{background:var(--accent-light);color:var(--accent)}
.toc-link.active{background:var(--accent);color:#fff;font-weight:600}

/* ── Content ─────────────────────────────────────── */
#content{max-width:52rem;margin:0 auto;padding:2rem 2.5rem 6rem}

/* ── Book cover ──────────────────────────────────── */
.book-cover{
  background:linear-gradient(135deg,var(--cover-from),var(--cover-to));
  color:#fff;padding:4rem 3rem;border-radius:12px;margin-bottom:3rem;
  text-align:center;
}
.book-cover h1{font-family:var(--serif);font-size:2.8rem;margin-bottom:.5rem;font-weight:700}
.book-cover .subtitle{font-size:1.2rem;opacity:.85;margin-bottom:1rem}
.book-cover .meta{font-size:.95rem;opacity:.7}

/* ── Chapter headings ────────────────────────────── */
.chapter{position:relative;padding-top:2.5rem;margin-bottom:3rem}
.chapter h1{
  font-family:var(--serif);font-size:2rem;color:var(--accent);
  border-bottom:3px solid var(--accent);padding-bottom:.5rem;
  margin-bottom:1.5rem;position:relative;
}
.chapter-watermark{
  position:absolute;right:0;top:-1.2rem;font-size:6rem;
  font-family:var(--serif);font-weight:700;
  opacity:.06;line-height:1;pointer-events:none;
}
.chapter h2{
  font-family:var(--serif);font-size:1.45rem;
  color:var(--accent);border-bottom:1px solid var(--border);
  padding-bottom:.3rem;margin:2rem 0 1rem;
}
.chapter h3{font-size:1.15rem;margin:1.5rem 0 .7rem;font-weight:600}
.chapter h4,.chapter h5,.chapter h6{font-size:1rem;margin:1.2rem 0 .5rem;font-weight:600}

/* ── Prose ───────────────────────────────────────── */
.chapter p{margin:.8rem 0}
.chapter ul,.chapter ol{margin:.8rem 0 .8rem 1.5rem}
.chapter li{margin:.3rem 0}
.chapter hr{border:none;border-top:1px solid var(--border);margin:2rem 0}

/* ── Inline code ─────────────────────────────────── */
.chapter code{
  font-family:var(--mono);font-size:.88em;
  background:var(--accent-light);padding:.15em .35em;border-radius:4px;
}

/* ── Code blocks ─────────────────────────────────── */
.code-wrapper{
  position:relative;background:var(--code-bg);border-radius:8px;
  margin:1rem 0;overflow:hidden;
}
.code-wrapper pre{
  padding:1.2rem 1.4rem;overflow-x:auto;margin:0;
}
.code-wrapper pre code{
  font-family:var(--mono);font-size:.85rem;line-height:1.55;
  color:var(--code-fg);background:transparent;padding:0;
}
.lang-badge{
  position:absolute;top:.5rem;right:.7rem;
  font-size:.7rem;font-family:var(--mono);text-transform:uppercase;
  color:#888;background:rgba(255,255,255,.08);
  padding:.15em .5em;border-radius:4px;
}

/* ── Math ────────────────────────────────────────── */
.math-block{
  background:var(--math-bg);padding:1.2rem 1.5rem;
  border-radius:8px;margin:1.2rem 0;overflow-x:auto;
  text-align:center;
}

/* ── Mermaid ─────────────────────────────────────── */
.mermaid-wrapper{
  background:var(--mermaid-bg);border:1px solid var(--border);
  border-radius:8px;padding:1.2rem;margin:1.2rem 0;overflow-x:auto;
  text-align:center;
}
.mermaid-wrapper pre.mermaid{margin:0;background:transparent}

/* ── Blockquote ──────────────────────────────────── */
.chapter blockquote{
  border-left:4px solid var(--blockquote-border);
  background:var(--blockquote-bg);padding:.8rem 1.2rem;
  border-radius:0 8px 8px 0;margin:1rem 0;font-style:italic;
}
.chapter blockquote p{margin:.3rem 0}

/* ── Tables ──────────────────────────────────────── */
.table-wrapper{overflow-x:auto;margin:1rem 0}
.chapter table{
  border-collapse:collapse;width:100%;font-size:.92rem;
}
.chapter th,.chapter td{
  border:1px solid var(--border);padding:.55rem .75rem;text-align:left;
}
.chapter th{background:var(--accent-light);font-weight:600}
.chapter tr:nth-child(even) td{background:var(--table-stripe)}

/* ── Exercise ────────────────────────────────────── */
.exercise{
  border-left:4px solid var(--exercise-border);
  background:var(--exercise-bg);padding:1rem 1.3rem;
  border-radius:0 8px 8px 0;margin:1.5rem 0;
}
.exercise::before{content:"🧪 ";font-size:1.1rem}

/* ── Images ──────────────────────────────────────── */
.chapter img{max-width:100%;border-radius:8px;margin:1rem 0}

/* ── Print ───────────────────────────────────────── */
@media print{
  #sidebar,#progress-bar{display:none!important}
  body{display:block;font-size:11pt}
  .book-cover{break-after:page}
  .chapter{break-inside:avoid-page}
  .code-wrapper{border:1px solid #ddd}
}
</style>
</head>
<body>
<div id="progress-bar"></div>

<nav id="sidebar">
  <h2>Contents</h2>
  __TOC__
</nav>

<main id="content">
  <div class="book-cover">
    <h1>nanochat</h1>
    <div class="subtitle">Understanding LLM Training from Scratch</div>
    <div class="meta">__CHAPTER_COUNT__ chapters &middot; A self-contained guide to the nanochat codebase</div>
  </div>

  __SECTIONS__
</main>

<!-- KaTeX (synchronous) -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<script>
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$',  right: '$',  display: false },
      { left: '\\(', right: '\\)', display: false },
      { left: '\\[', right: '\\]', display: true }
    ],
    throwOnError: false,
    ignoredTags: ['script','noscript','style','textarea','pre','code']
  });
</script>

<!-- Mermaid (ESM) -->
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  mermaid.initialize({
    startOnLoad: false,
    theme: window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'default',
    securityLevel: 'loose'
  });
  await mermaid.run();
</script>

<!-- Navigation & progress -->
<script>
(function(){
  // Progress bar
  const bar = document.getElementById('progress-bar');
  function updateProgress(){
    const h = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.width = h > 0 ? (window.scrollY / h * 100) + '%' : '0%';
  }
  window.addEventListener('scroll', updateProgress, {passive:true});
  updateProgress();

  // Active ToC highlighting
  const links = document.querySelectorAll('.toc-link');
  const sections = [];
  links.forEach(function(a){
    const id = a.getAttribute('data-chapter');
    const el = document.getElementById(id);
    if(el) sections.push({el:el, link:a});
  });

  function updateActive(){
    let current = null;
    const offset = window.scrollY + 120;
    for(let i = sections.length - 1; i >= 0; i--){
      if(sections[i].el.offsetTop <= offset){ current = i; break; }
    }
    links.forEach(function(a){ a.classList.remove('active'); });
    if(current !== null) sections[current].link.classList.add('active');
  }
  window.addEventListener('scroll', updateActive, {passive:true});
  updateActive();

  // Smooth scroll
  links.forEach(function(a){
    a.addEventListener('click', function(e){
      e.preventDefault();
      const id = a.getAttribute('data-chapter');
      const el = document.getElementById(id);
      if(el) el.scrollIntoView({behavior:'smooth', block:'start'});
    });
  });
})();
</script>
</body>
</html>
'@

# ── Inject content and write ─────────────────────────────────────────
$html = $template.Replace('__TOC__', $tocHtml)
$html = $html.Replace('__SECTIONS__', $sectionsHtml)
$html = $html.Replace('__CHAPTER_COUNT__', "$chapterCount")

# Write UTF-8 without BOM
$utf8noBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($outFile, $html, $utf8noBom)

$size = [math]::Round((Get-Item $outFile).Length / 1KB)
Write-Host "Built $outFile ($size KB, $chapterCount chapters)"
