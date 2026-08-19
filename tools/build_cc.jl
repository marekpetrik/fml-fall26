#!/usr/bin/env julia
#
# build_cc.jl — Generate a Canvas-importable IMS Common Cartridge (.imscc) for
# CS 550 whose items import as real Canvas *Assignments* (with due dates and
# assignment-group weights), not as modules or pages.
#
# Canvas only honours <due_at>/assignment_settings.xml when it recognises the
# package as a Canvas export, which requires the course_settings/ bundle (with
# the canvas_export.txt marker) and Canvas-style "g"+hex migration identifiers.
# module_meta.xml is left empty on purpose so that NO modules are created.
#
# Usage:  julia tools/build_cc.jl
# Output: fml-fall26.imscc in the repository root.
# Requires the `zip` command-line tool.

using Dates, SHA

# ---- paths ----------------------------------------------------------------
const REPO = dirname(@__DIR__)
const OUT  = joinpath(REPO, "fml-fall26.imscc")

const COURSE_TITLE = "CS 550: Foundations of Machine Learning (Fall 2026)"

# Assignment PDFs / sources are linked (not bundled) from the course website.
const PDF_BASE_URL = "https://gitlab.com/marekpetrik/fml-fall26/-/raw/main/assignments"

# ---- data -----------------------------------------------------------------
# (number, due date) — assignments due at 11:59 PM local (America/New_York).
const ASSIGNMENTS = [
    (1,  Date(2026, 9,  7)), (2,  Date(2026, 9, 14)), (3,  Date(2026, 9, 21)),
    (4,  Date(2026, 9, 28)), (5,  Date(2026, 10, 5)), (6,  Date(2026, 10, 12)),
    (7,  Date(2026, 10, 19)),(8,  Date(2026, 10, 26)),(9,  Date(2026, 11, 2)),
    (10, Date(2026, 11, 9)), (11, Date(2026, 11, 16)),(12, Date(2026, 11, 23)),
    (13, Date(2026, 11, 30)),
]
# (label, date) — in-class quizzes, due at the end of the 2:10–3:30 slot.
const QUIZZES = [
    ("01", Date(2026, 9, 10)), ("02", Date(2026, 9, 17)), ("03", Date(2026, 9, 24)),
    ("04", Date(2026, 10, 1)), ("05", Date(2026, 10, 8)), ("06", Date(2026, 10, 15)),
    ("07", Date(2026, 10, 22)),("08", Date(2026, 10, 29)),("09", Date(2026, 11, 5)),
    ("10", Date(2026, 11, 12)),("11", Date(2026, 11, 19)),("12", Date(2026, 12, 3)),
]
const ASSIGN_DUE_TIME = "T23:59:00"   # 11:59 PM
const QUIZ_DUE_TIME   = "T15:30:00"   # 3:30 PM, end of class

# Final exam — in-person, closed-book. UNH sets the date, so it defaults to
# `nothing` (imported with no due date). Set a Date once the slot is published.
const FINAL_EXAM = (title = "Final Exam", date = nothing, points = 100.0)
# In-person review of assignment solutions and code — a single graded item.
const REVIEW     = (title = "In-person Review", date = nothing, points = 20.0)

# Assignment groups and their weights (%), matching the syllabus grade breakdown.
# (key, title, weight %, drop-lowest count). Problem Sets drop the 3 worst.
const GROUPS = [
    (:homework, "Problem Sets",     10.0, 3),
    (:quizzes,  "Quizzes",          30.0, 0),
    (:review,   "In-person Review", 20.0, 0),
    (:final,    "Final Exam",       40.0, 0),
]

# ---- helpers --------------------------------------------------------------
# Canvas migration identifiers: "i" + 32 hex chars (matches Canvas's create_key,
# which is "i" + an MD5 hexdigest), derived deterministically from a seed.
make_id(seed) = "i" * bytes2hex(sha256(seed))[1:32]

pretty(d::Date) = Dates.format(d, "u d, yyyy")
pts(x)          = string(round(Float64(x); digits = 1))
xmlesc(s) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")

# US Eastern DST ends 2026-11-01; before that EDT (-04:00), on/after EST (-05:00).
tzoffset(d::Date) = d >= Date(2026, 11, 1) ? "-05:00" : "-04:00"
due_stamp(d::Date, t) = Dates.format(d, "yyyy-mm-dd") * t * tzoffset(d)

gid(key::Symbol) = make_id("group_$key")

# ---- reset build dir ------------------------------------------------------
build = mktempdir()
mkpath(joinpath(build, "course_settings"))

resources = Vector{Tuple{String,String,Vector{String}}}()   # (id, href, [files])

# ---- assignment_settings.xml ----------------------------------------------
function assignment_settings(id, group_id, title, due, all_day, points, subtypes, position)
    lines = String[
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<assignment identifier=\"$id\" xmlns=\"http://canvas.instructure.com/xsd/cccv1p0\" " *
            "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" " *
            "xsi:schemaLocation=\"http://canvas.instructure.com/xsd/cccv1p0\">",
        "  <title>$(xmlesc(title))</title>",
    ]
    if due !== nothing
        push!(lines, "  <due_at>$due</due_at>")
        all_day !== nothing && push!(lines, "  <all_day_date>$all_day</all_day_date>")
    end
    append!(lines, [
        "  <points_possible>$(pts(points))</points_possible>",
        "  <grading_type>points</grading_type>",
        "  <submission_types>$subtypes</submission_types>",
        "  <workflow_state>published</workflow_state>",
        "  <assignment_group_identifierref>$group_id</assignment_group_identifierref>",
        "  <position>$position</position>",
        "  <peer_reviews>false</peer_reviews>",
        "  <omit_from_final_grade>false</omit_from_final_grade>",
        "</assignment>",
        "",
    ])
    join(lines, "\n")
end

# create one assignment the way Canvas does: a folder named by the migration id,
# containing the HTML body and assignment_settings.xml. The manifest resource
# points its href at the HTML file, with assignment_settings.xml as a <file>.
function add_assignment(key, group, title, body; due = nothing, all_day = nothing,
                        points = 100.0, subtypes = "on_paper", position = 1)
    id     = make_id("res_$key")
    folder = joinpath(build, id)
    mkpath(folder)
    html_name = "$key.html"
    write(joinpath(folder, html_name), body)
    write(joinpath(folder, "assignment_settings.xml"),
          assignment_settings(id, gid(group), title, due, all_day, points, subtypes, position))
    push!(resources, (id, "$id/$html_name",
        ["$id/$html_name", "$id/assignment_settings.xml"]))
end

htmldoc(title, inner) = """
<html><head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"/><title>$(xmlesc(title))</title></head>
<body>
$inner
</body></html>"""

# ---- assignments ----------------------------------------------------------
for (i, (num, due)) in enumerate(ASSIGNMENTS)
    title = "Problem Set $num"
    body = htmldoc(title, """
    <h2>$title</h2>
    <p><strong>Due:</strong> $(pretty(due)) at 11:59&nbsp;PM.</p>
    <p>Turn in a <strong>PDF</strong> and a <strong>separate source code file</strong>. The problems are linked below:</p>
    <p><a href="$(PDF_BASE_URL)/assignment$num.pdf">Download Problem Set $num (PDF)</a></p>
    <p><a href="$(PDF_BASE_URL)/assignment$num.tex">Source for Problem Set $num (tex)</a></p>""")
    add_assignment("assignment$num", :homework, title, body;
        due = due_stamp(due, ASSIGN_DUE_TIME), all_day = Dates.format(due, "yyyy-mm-dd"),
        points = 100.0, subtypes = "online_upload,online_text_entry", position = i)
end

# ---- quizzes (in-class, on paper) ----------------------------------------
for (i, (label, due)) in enumerate(QUIZZES)
    title = "Quiz $label"
    body = htmldoc(title, """
    <h2>$title</h2>
    <p><strong>Date:</strong> $(pretty(due)), in class.</p>
    <p>In-class graded quiz. There is nothing to submit online.</p>""")
    add_assignment("quiz$label", :quizzes, title, body;
        due = due_stamp(due, QUIZ_DUE_TIME), all_day = Dates.format(due, "yyyy-mm-dd"),
        points = 10.0, subtypes = "on_paper", position = i)
end

# ---- in-person review -----------------------------------------------------
let rv = REVIEW
    due = rv.date === nothing ? nothing : due_stamp(rv.date, ASSIGN_DUE_TIME)
    when = rv.date === nothing ?
        "scheduled in class &mdash; see myCourses for the review schedule." : "$(pretty(rv.date))."
    body = htmldoc(rv.title, """
    <h2>$(rv.title)</h2>
    <p><strong>Format:</strong> in-person review of your assignment solutions and code.</p>
    <p><strong>When:</strong> $when Nothing to submit online.</p>""")
    add_assignment("in_person_review", :review, rv.title, body;
        due = due, points = rv.points, subtypes = "on_paper", position = 1)
end

# ---- final exam -----------------------------------------------------------
let fe = FINAL_EXAM
    due = fe.date === nothing ? nothing : due_stamp(fe.date, ASSIGN_DUE_TIME)
    when = fe.date === nothing ? "scheduled according to UNH policies." : "$(pretty(fe.date))."
    body = htmldoc(fe.title, """
    <h2>$(fe.title)</h2>
    <p><strong>Format:</strong> in-person, closed-book. No cheat sheet allowed.</p>
    <p><strong>Date:</strong> $when</p>""")
    add_assignment("final_exam", :final, fe.title, body;
        due = due, points = fe.points, subtypes = "on_paper", position = 1)
end

# ---- course_settings/ bundle (marks this as a Canvas export) --------------
cc = Dict{String,String}()
# Canvas's exact flavor-identifier marker (verbatim from Canvas's exporter).
cc["canvas_export.txt"] = "Q: What did the panda say when he was forced out of his natural habitat?\nA: This is un-BEAR-able\n"
cc["course_settings.xml"] = """
<?xml version="1.0" encoding="UTF-8"?>
<course identifier="$(make_id("course_settings"))" xmlns="http://canvas.instructure.com/xsd/cccv1p0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://canvas.instructure.com/xsd/cccv1p0">
  <title>$(xmlesc(COURSE_TITLE))</title>
  <course_code>CS 550</course_code>
  <grading_standard_enabled>false</grading_standard_enabled>
  <apply_assignment_group_weights>true</apply_assignment_group_weights>
  <workflow_state>available</workflow_state>
</course>
"""
# NB: Canvas's importer selects <assignmentGroup> (camelCase) nodes — snake_case
# silently yields zero groups. Leaf fields (group_weight, drop_type) are snake.
cc["assignment_groups.xml"] = string("""
<?xml version="1.0" encoding="UTF-8"?>
<assignmentGroups xmlns="http://canvas.instructure.com/xsd/cccv1p0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://canvas.instructure.com/xsd/cccv1p0">
""",
    join([begin
        rules = d > 0 ? string(
            "\n    <rules>\n      <rule>\n        <drop_type>drop_lowest</drop_type>",
            "\n        <drop_count>$d</drop_count>\n      </rule>\n    </rules>") : ""
        """  <assignmentGroup identifier="$(gid(k))">
    <title>$(xmlesc(t))</title>
    <position>$i</position>
    <group_weight>$(pts(w))</group_weight>$rules
  </assignmentGroup>"""
    end for (i, (k, t, w, d)) in enumerate(GROUPS)], "\n"),
    "\n</assignmentGroups>\n")
# empty module list => Canvas creates NO modules
cc["module_meta.xml"] = """
<?xml version="1.0" encoding="UTF-8"?>
<modules xmlns="http://canvas.instructure.com/xsd/cccv1p0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://canvas.instructure.com/xsd/cccv1p0">
</modules>
"""
cc["context_info.xml"] = """
<?xml version="1.0" encoding="UTF-8"?>
<context_info xmlns="http://canvas.instructure.com/xsd/cccv1p0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://canvas.instructure.com/xsd/cccv1p0">
  <course_name>$(xmlesc(COURSE_TITLE))</course_name>
</context_info>
"""
cc["files_meta.xml"] = """
<?xml version="1.0" encoding="UTF-8"?>
<fileMeta xmlns="http://canvas.instructure.com/xsd/cccv1p0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://canvas.instructure.com/xsd/cccv1p0">
</fileMeta>
"""
cc["media_tracks.xml"] = """
<?xml version="1.0" encoding="UTF-8"?>
<media_tracks xmlns="http://canvas.instructure.com/xsd/cccv1p0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://canvas.instructure.com/xsd/cccv1p0">
</media_tracks>
"""
for (name, content) in cc
    write(joinpath(build, "course_settings", name), content)
end
cc_files = ["course_settings/$name" for name in sort(collect(keys(cc)))]
cc_res_id = make_id("course_settings_res")

# ---- imsmanifest.xml ------------------------------------------------------
res_xml = String[]
# course settings bundle first (no intendeduse)
push!(res_xml, string(
    "    <resource identifier=\"$cc_res_id\" type=\"associatedcontent/imscc_xmlv1p1/learning-application-resource\" href=\"course_settings/canvas_export.txt\">\n",
    join(["      <file href=\"$f\"/>" for f in cc_files], "\n"),
    "\n    </resource>"))
# then every assignment
for (id, href, files) in resources
    fx = join(["      <file href=\"$f\"/>" for f in files], "\n")
    push!(res_xml, "    <resource identifier=\"$id\" type=\"associatedcontent/imscc_xmlv1p1/learning-application-resource\" href=\"$href\">\n$fx\n    </resource>")
end

manifest = """
<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="$(make_id("course"))"
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
        <lomimscc:title><lomimscc:string>$(xmlesc(COURSE_TITLE))</lomimscc:string></lomimscc:title>
      </lomimscc:general>
    </lomimscc:lom>
  </metadata>
  <organizations>
    <organization identifier="$(make_id("org"))" structure="rooted-hierarchy">
      <item identifier="$(make_id("root"))"/>
    </organization>
  </organizations>
  <resources>
$(join(res_xml, "\n"))
  </resources>
</manifest>
"""
write(joinpath(build, "imsmanifest.xml"), manifest)

# ---- zip (STORED, manifest first) -----------------------------------------
isfile(OUT) && rm(OUT)
cd(build) do
    run(`zip -0 -X -q $OUT imsmanifest.xml`)
    run(`zip -0 -X -q -r $OUT . -x imsmanifest.xml`)
end
println("Wrote ", OUT)
println("Assignments: ", length(ASSIGNMENTS), "  Quizzes: ", length(QUIZZES),
        "  Review: 1  Final exam: 1  Groups: ", length(GROUPS),
        "  Modules: 0")
