"""
Display and table-styling helpers for the ML vs. GLM thesis notebook.

These functions are purely presentational (consistent table styling,
section headings, and notice boxes) and are kept separate from metrics.py,
which contains the actual statistical methodology, so the two files can be
read and reasoned about independently.
"""

from IPython.display import display, HTML


def style_table(df, caption=None, formats=None, index=False, hide_cols=None):
    """
    Apply consistent, thesis-appropriate styling to a results table: dark
    header row, zebra striping, formatted numbers, and (by default) a
    hidden row index, matching the style used for the raw data preview.
    """
    styler = df.style.set_table_styles([
        {'selector': 'caption', 'props': [('font-size', '13px'), ('font-weight', 'bold'),
                                           ('caption-side', 'top'), ('text-align', 'left'),
                                           ('padding-bottom', '8px'), ('color', '#222')]},
        {'selector': 'th', 'props': [('background-color', '#2c3e50'), ('color', 'white'),
                                      ('text-align', 'center'), ('padding', '6px 12px'),
                                      ('font-size', '12px')]},
        {'selector': 'td', 'props': [('padding', '5px 12px'), ('text-align', 'center'),
                                      ('font-size', '12px'), ('border-bottom', '1px solid #ddd')]},
        {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f7f9fb')]},
    ])
    if caption:
        styler = styler.set_caption(caption)
    if formats:
        styler = styler.format(formats)
    if hide_cols:
        styler = styler.hide(axis='columns', subset=hide_cols)
    if not index:
        styler = styler.hide(axis='index')
    return styler


def section_heading(text):
    """Render a consistent section/sub-section heading above a table or figure."""
    display(HTML(
        f"<div style='font-size:15px; font-weight:bold; color:#222; "
        f"padding:10px 0 8px 0;'>{text}</div>"
    ))


def info_box(html_content):
    """Render a consistent left-bordered info/notice box (e.g. 'Saved to: ...'
    notices, or short descriptive notes) below a table or figure."""
    display(HTML(
        f"""
        <div style="
            margin-top:10px;
            font-size:12px;
            color:#555;
            padding:6px 12px;
            border-left:3px solid #2c3e50;
            background-color:#f7f9fb;
        ">
            {html_content}
        </div>
        """
    ))


def saved_notice(*paths):
    """Shorthand info_box for 'Saved to: <path>' notices, supporting multiple paths."""
    sep = "&nbsp;&nbsp;|&nbsp;&nbsp;"
    info_box("<b>Saved to:</b> " + sep.join(str(p) for p in paths))
