---
title: Cognitive graph landing-zone handoff
type: guide
status: stable
updated: 2026-10-13
---

# Cognitive graph landing-zone handoff

The graph and its scientific scope are maintained here; reusable submission and
upload methods are maintained in [daf-okn](https://github.com/docxology/daf-okn).
Registry acceptance, credentials, upload, processing and hosting require separate
receipts. Local package preparation does not establish any of those events.

## Prepare and verify

Regenerate the canonical graph and check its metadata, then use daf-okn to
package those exact bytes with attribution, the registry entry and example queries.
Run the latter commands from an environment containing the updated methods.

```bash
uv run --extra dev python -m scripts.export_rdf
uv run --extra dev python -m scripts.export_rdf --check-submission
daf-okn bundle-rdf --root . --output output/okn-submission
daf-okn bundle-rdf --output output/okn-submission --check
```

The bundle's `registry/` directory uses the upstream registry layout. Query files
contain the [required YAML metadata headers](https://registry.okn.us/help/update-queries/).
The cross-graph query can run server-side once the local graph is hosted; its
current evidence comes from the local export and captured remote responses.

## Landing-zone sequence

1. Obtain registry acceptance and lakeFS access for repository `cognitive-kb`.
   The [setup guide](https://registry.okn.us/help/setup/) directs credential
   requests to Yaphet Kebede. No lakeFS credential environment or configuration
   was available in this session; an authenticated upload was not attempted.
2. Verify `bundle.json`, then upload only `cognitive-kb.nq` to
   `s3://cognitive-kb/develop/cognitive-kb.nq` using the documented endpoint
   `https://repository.okn.us/`. daf-okn's `upload-commands` method generates
   commands for the selected client. Keep JSON manifests, documentation and
   raw remote evidence outside the RDF ingestion path.
3. Commit the lakeFS develop change and retain the object hash and commit ID.
   A copied object is not a committed update.
4. Follow the [update workflow](https://registry.okn.us/help/update/) for the
   develop-to-main merge, conversion, stable output and deployment tag. Retain
   processing logs, stable graph identifier, version tag and any failures.
5. Verify deployed dataset licensing, named graph, basic counts and example
   queries. Re-run `cross_graph_obo.rq` on the federation endpoint. RENCI adds
   the `sparql` and `tpf` registry fields once the endpoints exist.

## Access-request draft

To: Yaphet Kebede, kebedey@renci.org

Subject: lakeFS access for cognitive-kb

Hello Yaphet,

We are preparing the Active Inference Institute's cognitive-kb graph for OKN.
The registry entry is submitted in frink-okn/okn-registry PR 535. Repository-owned
knowledge-base content and exports are CC BY 4.0. We have also captured an actual
identifier-based join from our concept pages through identifier-mappings to
Ubergraph labels, with replayable queries and response hashes.

Could you provide the supported credential provisioning process and access to
the cognitive-kb lakeFS repository's develop branch? Please also confirm any
remaining registry or data-review requirements before the initial upload.

Graph source: https://github.com/ActiveInferenceInstitute/cognitive

Reusable methods: https://github.com/docxology/daf-okn

Thank you,
Daniel Ari Friedman

This is an unsent draft. Credential provisioning and RENCI's acceptance decision
are the remaining external prerequisites for an upload.
