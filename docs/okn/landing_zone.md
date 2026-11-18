---
title: Cognitive graph landing-zone handoff
type: guide
status: stable
updated: 2026-11-17
---

# Cognitive graph landing-zone handoff

The initial registry entry was merged on November 3, 2026. RENCI provisioned
access to the [cognitive-kb lakeFS repository](https://repository.okn.us/repositories/cognitive-kb/objects).
See the [registry pull request](https://github.com/frink-okn/okn-registry/pull/535)
and the [proposed description update](registry_entry.md) for the current graph model.

This repository produces the canonical RDF. The
[daf-okn project](https://github.com/docxology/daf-okn) packages that export and
handles lakeFS preflight, staging and verification.

## Prepare the release bundle

Start from a clean cognitive checkout at the reviewed source commit. Run the
submission and graph quality checks, regenerate the RDF, then use a daf-okn
revision that provides `lakefs-cognitive` to package and verify the export.
Choose a new bundle directory for each candidate.

```bash
./run.sh docs
uv run --locked --extra dev --python 3.12 python -m scripts.export_rdf
daf-okn bundle-rdf --root . --output output/okn-submission
daf-okn bundle-rdf --output output/okn-submission --check
```

`bundle.json` records the source commit and file hashes. The bundle includes an
attribution notice and the `registry/` submission tree, including queries with
the upstream [YAML metadata headers](https://registry.okn.us/help/update-queries/).
The ingestion object is the root `cognitive-kb.nq` file. Keep the Turtle export,
JSON manifests, documentation and captured remote responses with the local bundle.

## Verify access and stage on develop

Keep the RENCI lakectl YAML outside Git in a user-private location. Set
`LAKECTL_CONFIG_FILE` to its path. Preflight checks the candidate and reads the
remote repository before staging.

```bash
daf-okn lakefs-cognitive --bundle output/okn-submission --config "$LAKECTL_CONFIG_FILE"
```

Once the preflight reports a clean `develop` branch, stage the reviewed bundle.
The staging command uploads and commits `cognitive-kb.nq` on `develop`. Save the
JSON receipt with the bundle and verify the remote bytes again at the lakeFS
commit ID reported by the command.

```bash
daf-okn lakefs-cognitive --bundle output/okn-submission --config "$LAKECTL_CONFIG_FILE" --stage
daf-okn lakefs-cognitive --bundle output/okn-submission --config "$LAKECTL_CONFIG_FILE" --commit "$LAKEFS_COMMIT_ID"
```

## Processing and deployment

1. Compare `develop` with protected `main` and review the update before opening
   a lakeFS merge request. Merging starts conversion.
2. Confirm that conversion produced the expected `stable_v*` branch. Follow
   the [OKN update workflow](https://registry.okn.us/help/update/) for tagging and
   deployment. Keep the source commit, lakeFS object hash and commit ID,
   processing result, stable branch and deployment tag together.
3. Verify the deployed dataset's license, named graph and example queries.
   Re-run `cross_graph_obo.rq` on the federation endpoint. RENCI adds the
   `sparql` and `tpf` registry fields when the endpoints are ready.
