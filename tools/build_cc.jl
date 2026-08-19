#!/usr/bin/env julia
#
# build_cc.jl — Generate a Canvas-compatible IMS Common Cartridge (.imscc) for
# CS 550 that contains the assignments and the in-class quizzes, each with its
# due date (Canvas <due_at>, so the dates import as real dates in myCourses).
#
# Usage:  julia tools/build_cc.jl
# Output: fml-fall26.imscc in the repository root.
#
# Requires the `zip` command-line tool (used to assemble the package). The
# assignment PDFs must already be compiled in assignments/assignmentN.pdf.

using Dates

# ---- paths ----------------------------------------------------------------
const REPO  = dirname(@__DIR__)                       # repo root (tools/ is under it)
const OUT   = joinpath(REPO, "fml-fall26.imscc")

# ---- data -----------------------------------------------------------------
# (number, due date) — assignments are due at 11:59 PM.
const ASSIGNMENTS = [
    (1,  Date(2026, 9,  7)), (2,  Date(2026, 9, 14)), (3,  Date(2026, 9, 21)),
    (4,  Date(2026, 9, 28)), (5,  Date(2026, 10, 5)), (6,  Date(2026, 10, 12)),
    (7,  Date(2026, 10, 19)),(8,  Date(2026, 10, 26)),(9,  Date(2026, 11, 2)),
    (10, Date(2026, 11, 9)), (11, Date(2026, 11, 16)),(12, Date(2026, 11, 23)),
    (13, Date(2026, 11, 30)),
]
# (label, date) — in-class quizzes, "due" at the end of the 2:10–3:30 slot.
const QUIZZES = [
    ("01", Date(2026, 9, 10)), ("02", Date(2026, 9, 17)), ("03", Date(2026, 9, 24)),
    ("04", Date(2026, 10, 1)), ("05", Date(2026, 10, 8)), ("06", Date(2026, 10, 15)),
    ("07", Date(2026, 10, 22)),("08", Date(2026, 10, 29)),("09", Date(2026, 11, 5)),
    ("10", Date(2026, 11, 12)),("11", Date(2026, 11, 19)),("12", Date(2026, 12, 3)),
]
const ASSIGN_DUE_TIME = "T23:59:00"   # 11:59 PM
const QUIZ_DUE_TIME   = "T15:30:00"   # 3:30 PM, end of class

# Final exam — in-person, closed-book, 40% of the grade. UNH schedules the exact
# date, so `date` defaults to `nothing` (imported with no due date). Set it to a
# `Date(2026, 12, ...)` once the registrar publishes the exam slot.
const FINAL_EXAM = (title = "Final Exam", date = nothing, points = 100.0)

pretty(d::Date) = Dates.format(d, "u d, yyyy")             # e.g. "Sep 7, 2026"
duestamp(d, t)  = Dates.format(d, "yyyy-mm-dd") * t         # e.g. "2026-09-07T23:59:00"

# minimal XML text escaping
xmlesc(s) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;",
                       "\"" => "&quot;")

# ---- reset build dir ------------------------------------------------------
build = mktempdir()
mkpath(joinpath(build, "web_resources", "assignments"))

# resources:    (identifier, type, href, [files])
# file_resources: (identifier, relpath)
resources      = Vector{Tuple{String,String,String,Vector{String}}}()
file_resources = Vector{Tuple{String,String}}()
assign_items   = Vector{Tuple{String,String,String}}()   # (item id, ref, title)
quiz_items     = Vector{Tuple{String,String,String}}()
exam_items     = Vector{Tuple{String,String,String}}()

# `due` may be a timestamp string or `nothing` (omits <due_at> entirely).
function assignment_settings(ident, title, due, points, subtypes)
    lines = String[
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<assignment xmlns=\"http://canvas.instructure.com/xsd/cccv1p0\" identifier=\"$ident\">",
        "  <title>$(xmlesc(title))</title>",
    ]
    due === nothing || push!(lines, "  <due_at>$due</due_at>")
    append!(lines, [
        "  <points_possible>$(Printf_fixed(points))</points_possible>",
        "  <grading_type>points</grading_type>",
        "  <submission_types>$subtypes</submission_types>",
        "  <workflow_state>published</workflow_state>",
        "</assignment>",
        "",
    ])
    join(lines, "\n")
end

# format a float with one decimal (e.g. 100.0) without pulling in Printf macros
Printf_fixed(x) = string(round(x; digits = 1))

# ---- assignments ----------------------------------------------------------
for (num, due) in ASSIGNMENTS
    ident  = "assignment$num"
    folder = joinpath(build, ident)
    mkpath(folder)
    due_dt = duestamp(due, ASSIGN_DUE_TIME)
    title  = "Problem Set $num"

    # copy the PDF and the LaTeX source into course files (web_resources) so
    # both can be linked from the assignment description
    pdf_rel = "web_resources/assignments/assignment$num.pdf"
    cp(joinpath(REPO, "assignments", "assignment$num.pdf"),
       joinpath(build, pdf_rel); force = true)
    push!(file_resources, ("file_assignment$(num)_pdf", pdf_rel))

    tex_rel = "web_resources/assignments/assignment$num.tex"
    cp(joinpath(REPO, "assignments", "assignment$num.tex"),
       joinpath(build, tex_rel); force = true)
    push!(file_resources, ("file_assignment$(num)_tex", tex_rel))

    body = """
    <html><head><meta charset="utf-8"><title>$title</title></head><body>
    <h2>$title</h2>
    <p><strong>Due:</strong> $(pretty(due)) at 11:59&nbsp;PM.</p>
    <p>Turn in a <strong>PDF</strong> and a <strong>separate source code file</strong>.
    The problems are in the attached PDF:</p>
    <p><a href="\$IMS-CC-FILEBASE\$/assignments/assignment$num.pdf">Download Problem Set $num (PDF)</a></p>
    <p><a href="\$IMS-CC-FILEBASE\$/assignments/assignment$num.tex">Source for Problem Set $num (tex)</a></p>
    </body></html>"""
    html_name = "$ident.html"
    write(joinpath(folder, html_name), body)
    write(joinpath(folder, "assignment_settings.xml"),
          assignment_settings(ident, title, due_dt, 100.0,
                              "online_upload,online_text_entry"))

    push!(resources, (ident,
        "associatedcontent/imscc_xmlv1p1/learning-application-resource",
        "$ident/$html_name",
        ["$ident/$html_name", "$ident/assignment_settings.xml"]))
    push!(assign_items, ("item_$ident", ident,
                         "Problem Set $num (due $(pretty(due)))"))
end

# ---- quizzes (in-class, on paper) ----------------------------------------
for (label, due) in QUIZZES
    ident  = "quiz$label"
    folder = joinpath(build, ident)
    mkpath(folder)
    due_dt = duestamp(due, QUIZ_DUE_TIME)
    title  = "Quiz $label"

    body = """
    <html><head><meta charset="utf-8"><title>$title</title></head><body>
    <h2>$title</h2>
    <p><strong>Date:</strong> $(pretty(due)), in class.</p>
    <p>In-class graded quiz. There is nothing to submit online.</p>
    </body></html>"""
    html_name = "$ident.html"
    write(joinpath(folder, html_name), body)
    write(joinpath(folder, "assignment_settings.xml"),
          assignment_settings(ident, title, due_dt, 10.0, "on_paper"))

    push!(resources, (ident,
        "associatedcontent/imscc_xmlv1p1/learning-application-resource",
        "$ident/$html_name",
        ["$ident/$html_name", "$ident/assignment_settings.xml"]))
    push!(quiz_items, ("item_$ident", ident,
                       "Quiz $label (due $(pretty(due)))"))
end

# ---- final exam (in-person, closed-book) ---------------------------------
let ident = "final_exam", fe = FINAL_EXAM
    folder = joinpath(build, ident)
    mkpath(folder)
    due_dt   = fe.date === nothing ? nothing : duestamp(fe.date, ASSIGN_DUE_TIME)
    date_txt = fe.date === nothing ?
        "scheduled according to UNH policies &mdash; see myCourses for the exact date and time." :
        "$(pretty(fe.date))."

    body = """
    <html><head><meta charset="utf-8"><title>$(fe.title)</title></head><body>
    <h2>$(fe.title)</h2>
    <p><strong>Format:</strong> in-person, closed-book. No cheat sheet allowed.</p>
    <p><strong>Date:</strong> $date_txt</p>
    </body></html>"""
    html_name = "$ident.html"
    write(joinpath(folder, html_name), body)
    write(joinpath(folder, "assignment_settings.xml"),
          assignment_settings(ident, fe.title, due_dt, fe.points, "on_paper"))

    push!(resources, (ident,
        "associatedcontent/imscc_xmlv1p1/learning-application-resource",
        "$ident/$html_name",
        ["$ident/$html_name", "$ident/assignment_settings.xml"]))
    push!(exam_items, ("item_$ident", ident,
        fe.date === nothing ? fe.title : "$(fe.title) ($(pretty(fe.date)))"))
end

# ---- manifest -------------------------------------------------------------
function items_xml(items)
    join(["""          <item identifier="$id" identifierref="$ref">
            <title>$(xmlesc(title))</title>
          </item>""" for (id, ref, title) in items], "\n")
end

res_xml = String[]
for (ident, rtype, href, files) in resources
    fx = join(["      <file href=\"$f\"/>" for f in files], "\n")
    push!(res_xml, """    <resource identifier="$ident" type="$rtype" href="$href">
$fx
    </resource>""")
end
for (fid, rel) in file_resources
    push!(res_xml, """    <resource identifier="$fid" type="webcontent" href="$rel">
      <file href="$rel"/>
    </resource>""")
end

manifest = """
<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="fml-fall26"
  xmlns="http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1"
  xmlns:lomimscc="http://ltsc.ieee.org/xsd/imsccv1p1/LOM/manifest"
  xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p1/LOM/resource"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1 http://www.imsglobal.org/profile/cc/ccv1p1/ccv1p1_imscp_v1p2_v1p0.xsd http://ltsc.ieee.org/xsd/imsccv1p1/LOM/manifest http://www.imsglobal.org/profile/cc/ccv1p1/LOM/ccv1p1_lommanifest_v1p0.xsd http://ltsc.ieee.org/xsd/imsccv1p1/LOM/resource http://www.imsglobal.org/profile/cc/ccv1p1/LOM/ccv1p1_lomresource_v1p0.xsd">
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.1.0</schemaversion>
    <lomimscc:lom>
      <lomimscc:general>
        <lomimscc:title><lomimscc:string>CS 550: Foundations of Machine Learning (Fall 2026)</lomimscc:string></lomimscc:title>
      </lomimscc:general>
    </lomimscc:lom>
  </metadata>
  <organizations>
    <organization identifier="org1" structure="rooted-hierarchy">
      <item identifier="root">
        <item identifier="mod_assignments">
          <title>Assignments</title>
$(items_xml(assign_items))
        </item>
        <item identifier="mod_quizzes">
          <title>In-class Graded Quizzes</title>
$(items_xml(quiz_items))
        </item>
        <item identifier="mod_final">
          <title>Final Exam</title>
$(items_xml(exam_items))
        </item>
      </item>
    </organization>
  </organizations>
  <resources>
$(join(res_xml, "\n"))
  </resources>
</manifest>
"""
write(joinpath(build, "imsmanifest.xml"), manifest)

# ---- zip ------------------------------------------------------------------
isfile(OUT) && rm(OUT)
cd(build) do
    run(`zip -qr $OUT .`)
end
println("Wrote ", OUT)
println("Assignments: ", length(ASSIGNMENTS), "  Quizzes: ", length(QUIZZES),
        "  Final exam: 1", "  Resources: ", length(resources) + length(file_resources))
