# Publication handoff

The technical pilot review, reporting correction, derived tables, reproducibility
manifest, response examples, two figures and first blog draft are complete.
Original artifacts are preserved. No further GPU run is needed for the bounded
pilot article. The full experiment is explicitly unfinished.

## Author review before publishing

- Review the first-person draft for your voice and intended audience. No biography,
  affiliation, dollar cost or personal reaction has been invented.
- Keep the scope: ten unique synthetic cases, 18 selected cells, 180 measured
  responses, one whole-cell rerun with a changed timeout. There are not 180
  independent semantic cases or repeated statistical trials.
- Keep decision-only and strict citation scores separate. A successful task means
  transport/runtime success, not correct answers. No threshold was relaxed to
  turn a failed experiment into a positive quality result.
- Keep the negative findings: 36 strict matches, 63 decision/limit matches, 18
  contexts missing required rules, and the full-cell time-limit problem.
- Review the example responses against their expected decisions and source rules.
  Reasons are not semantically scored. Rule coverage is a presence diagnostic,
  not proof of reasoning or minimum context size.
- If you recover exact built-image identities, add the Cloud Run metadata as a
  provenance supplement. Until then retain the explicit image-digest limitation;
  a pinned base image is not proof of the complete deployed image.
- Choose your publication platform. Upload the supplied PNG or SVG figures and
  replace relative image links with the hosted asset URLs. Link the study's
  repository files for evidence, rather than exposing private GCS objects.
- Remove the draft-status note only after your review. Check the rendered tables,
  figure captions and mobile layout. Publishing the post is intentionally left
  to you; no blogging service has been updated.

## Optional editorial handoff to another ChatGPT chat

Upload `blog-draft.md`, `FINAL_RESULTS.md`, `quality-review.md`, `results/cells.csv`
and the figures. Suggested instruction:

> Edit this pilot-study draft for clarity and voice. Preserve every numerical
> claim, the distinction between decision-only and citation correctness, the
> small synthetic workload, and all reproducibility limitations. Do not invent
> GPU runs, costs, statistical significance or production conclusions. Treat
> the four-hour full-cell duration as an extrapolation. Keep the 600-second rerun
> distinct from the original failed attempt. Suggest editorial changes for review.

## Future research, separate from this closeout

Use more held-out cases and independent repetitions, measure host-memory peaks
and actual billed resources, check actual tokenizer counts for the largest input,
and decide the full enterprise dataset. A full cell exceeding one hour needs a
longer-lived GPU platform or an explicitly revised batching/cache protocol. Any
model, prompt, scoring or workload change starts a new experimental condition.
