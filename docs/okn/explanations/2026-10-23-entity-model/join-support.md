# Join assertion-path support

Assertion paths add no composed mapping, identity, class equivalence, membership, reasoning, remote review, scientific or human validation. Hashes do not authenticate authorship. Capture time does not establish freshness or review time. Local blank nodes/lexical normalization are unsupported. Duplicate scalar rows share all typed solutions; individual language assignments cannot be recovered.

Captured observation: 2026-10-23T17:31:45.038105+00:00

The JSON preserves typed assertions, complete local reviews and raw binding origins.

```json
{
  "curation_decisions": "not_supplied",
  "deprecation_replacement": "not_captured",
  "explanation_schema_version": 1,
  "groups": [
    {
      "row_indices": [
        0
      ],
      "row_sha256": "33cd8da33f980cf2b795bbbce6b7ed4cc85548f5f8651d31c8e8907fc04eed15",
      "solutions": [
        {
          "bindings": {
            "concept": {
              "type": "uri",
              "value": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/cortisol"
            },
            "external": {
              "type": "uri",
              "value": "http://purl.obolibrary.org/obo/CHEBI_17650"
            },
            "externalLabel": {
              "type": "literal",
              "value": "cortisol"
            },
            "label": {
              "type": "literal",
              "value": "Cortisol",
              "xml:lang": "en"
            },
            "page": {
              "type": "uri",
              "value": "https://purl.org/okn/frink/kg/cognitive-kb/page/biology/cortisol"
            },
            "source": {
              "type": "uri",
              "value": "https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/cortisol.md"
            },
            "wikidata": {
              "type": "uri",
              "value": "http://www.wikidata.org/entity/Q190875"
            }
          },
          "local": {
            "assertions": [
              {
                "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                "o": {
                  "type": "literal",
                  "value": "Cortisol",
                  "xml:lang": "en"
                },
                "p": {
                  "type": "uri",
                  "value": "http://www.w3.org/2004/02/skos/core#prefLabel"
                },
                "s": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/cortisol"
                }
              },
              {
                "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                "o": {
                  "type": "uri",
                  "value": "http://www.w3.org/2004/02/skos/core#Concept"
                },
                "p": {
                  "type": "uri",
                  "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                },
                "s": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/cortisol"
                }
              },
              {
                "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                "o": {
                  "type": "uri",
                  "value": "https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/cortisol.md"
                },
                "p": {
                  "type": "uri",
                  "value": "http://purl.org/dc/terms/source"
                },
                "s": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/page/biology/cortisol"
                }
              },
              {
                "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                "o": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/cortisol"
                },
                "p": {
                  "type": "uri",
                  "value": "http://purl.org/dc/terms/subject"
                },
                "s": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/page/biology/cortisol"
                }
              },
              {
                "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                "o": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/cortisol"
                },
                "p": {
                  "type": "uri",
                  "value": "https://schema.org/about"
                },
                "s": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/page/biology/cortisol"
                }
              },
              {
                "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                "o": {
                  "type": "uri",
                  "value": "https://schema.org/CreativeWork"
                },
                "p": {
                  "type": "uri",
                  "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                },
                "s": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/page/biology/cortisol"
                }
              }
            ],
            "mappings": [
              {
                "assertion": {
                  "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                  "o": {
                    "type": "uri",
                    "value": "http://www.wikidata.org/entity/Q190875"
                  },
                  "p": {
                    "type": "uri",
                    "value": "http://www.w3.org/2004/02/skos/core#exactMatch"
                  },
                  "s": {
                    "type": "uri",
                    "value": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/cortisol"
                  }
                },
                "reviews": [
                  {
                    "assertions": [
                      {
                        "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                        "o": {
                          "datatype": "http://www.w3.org/2001/XMLSchema#date",
                          "type": "literal",
                          "value": "2026-10-20"
                        },
                        "p": {
                          "type": "uri",
                          "value": "http://purl.org/dc/terms/date"
                        },
                        "s": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ddec7991a1189fc969710af711930353fd57001a0a400fda6f8308c220ae8fe9"
                        }
                      },
                      {
                        "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                        "o": {
                          "type": "literal",
                          "value": "The page identifies the glucocorticoid steroid hormone cortisol, the same substance denoted by the target. The assertion concerns the concept, not identity of the Markdown document with a molecule.",
                          "xml:lang": "en"
                        },
                        "p": {
                          "type": "uri",
                          "value": "http://purl.org/dc/terms/description"
                        },
                        "s": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ddec7991a1189fc969710af711930353fd57001a0a400fda6f8308c220ae8fe9"
                        }
                      },
                      {
                        "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                        "o": {
                          "type": "uri",
                          "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement"
                        },
                        "p": {
                          "type": "uri",
                          "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                        },
                        "s": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ddec7991a1189fc969710af711930353fd57001a0a400fda6f8308c220ae8fe9"
                        }
                      },
                      {
                        "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                        "o": {
                          "type": "uri",
                          "value": "http://www.w3.org/2004/02/skos/core#exactMatch"
                        },
                        "p": {
                          "type": "uri",
                          "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"
                        },
                        "s": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ddec7991a1189fc969710af711930353fd57001a0a400fda6f8308c220ae8fe9"
                        }
                      },
                      {
                        "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                        "o": {
                          "type": "uri",
                          "value": "http://www.wikidata.org/entity/Q190875"
                        },
                        "p": {
                          "type": "uri",
                          "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#object"
                        },
                        "s": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ddec7991a1189fc969710af711930353fd57001a0a400fda6f8308c220ae8fe9"
                        }
                      },
                      {
                        "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                        "o": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/cortisol"
                        },
                        "p": {
                          "type": "uri",
                          "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"
                        },
                        "s": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ddec7991a1189fc969710af711930353fd57001a0a400fda6f8308c220ae8fe9"
                        }
                      },
                      {
                        "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                        "o": {
                          "type": "uri",
                          "value": "https://www.wikidata.org/wiki/Q190875"
                        },
                        "p": {
                          "type": "uri",
                          "value": "http://purl.org/dc/terms/source"
                        },
                        "s": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ddec7991a1189fc969710af711930353fd57001a0a400fda6f8308c220ae8fe9"
                        }
                      }
                    ],
                    "statement": {
                      "type": "uri",
                      "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ddec7991a1189fc969710af711930353fd57001a0a400fda6f8308c220ae8fe9"
                    }
                  }
                ]
              }
            ],
            "origin": {
              "local_sha256": "eb49ccbb6f729f61807912170c30f9a258d700c40c1a81f8f341361f6683049e"
            }
          },
          "multiplicity": 1,
          "remote_label": {
            "assertion": {
              "graph": "https://purl.org/okn/frink/kg/ubergraph",
              "o": {
                "type": "literal",
                "value": "cortisol"
              },
              "p": {
                "type": "uri",
                "value": "http://www.w3.org/2000/01/rdf-schema#label"
              },
              "s": {
                "type": "uri",
                "value": "http://purl.obolibrary.org/obo/CHEBI_17650"
              }
            },
            "occurrences": [
              {
                "binding_index": 0,
                "capture_index": 1,
                "graph": "https://purl.org/okn/frink/kg/ubergraph",
                "json_pointer": "/results/bindings/0",
                "lexical_or_metadata_changed": false,
                "materialized": {
                  "graph": "https://purl.org/okn/frink/kg/ubergraph",
                  "o": {
                    "type": "literal",
                    "value": "cortisol"
                  },
                  "p": {
                    "type": "uri",
                    "value": "http://www.w3.org/2000/01/rdf-schema#label"
                  },
                  "s": {
                    "type": "uri",
                    "value": "http://purl.obolibrary.org/obo/CHEBI_17650"
                  }
                },
                "query_file": "ubergraph-0000.rq",
                "query_sha256": "4152aa8c03db362b2e13d0c61dec7efcbbc1e26d9e391d3a10f10dbe7bb2bc70",
                "raw": {
                  "o": {
                    "type": "literal",
                    "value": "cortisol"
                  },
                  "p": {
                    "type": "uri",
                    "value": "http://www.w3.org/2000/01/rdf-schema#label"
                  },
                  "s": {
                    "type": "uri",
                    "value": "http://purl.obolibrary.org/obo/CHEBI_17650"
                  }
                },
                "response_file": "ubergraph-0000.json",
                "response_sha256": "346a9ffffa37a0b05ed7724588f2796a9b99a3f4625d9b06398b5fbbac757289"
              }
            ]
          },
          "remote_mapping": {
            "assertion": {
              "graph": "https://purl.org/okn/frink/kg/identifier-mappings",
              "o": {
                "type": "uri",
                "value": "http://purl.obolibrary.org/obo/CHEBI_17650"
              },
              "p": {
                "type": "uri",
                "value": "http://www.w3.org/2004/02/skos/core#exactMatch"
              },
              "s": {
                "type": "uri",
                "value": "http://www.wikidata.org/entity/Q190875"
              }
            },
            "occurrences": [
              {
                "binding_index": 1,
                "capture_index": 0,
                "graph": "https://purl.org/okn/frink/kg/identifier-mappings",
                "json_pointer": "/results/bindings/1",
                "lexical_or_metadata_changed": false,
                "materialized": {
                  "graph": "https://purl.org/okn/frink/kg/identifier-mappings",
                  "o": {
                    "type": "uri",
                    "value": "http://purl.obolibrary.org/obo/CHEBI_17650"
                  },
                  "p": {
                    "type": "uri",
                    "value": "http://www.w3.org/2004/02/skos/core#exactMatch"
                  },
                  "s": {
                    "type": "uri",
                    "value": "http://www.wikidata.org/entity/Q190875"
                  }
                },
                "query_file": "identifier-mappings-0000.rq",
                "query_sha256": "59b5476822c5ffbc27b588aa7fb5bcda53b776853f06d31d66ac9756b4ef11d6",
                "raw": {
                  "o": {
                    "type": "uri",
                    "value": "http://purl.obolibrary.org/obo/CHEBI_17650"
                  },
                  "p": {
                    "type": "uri",
                    "value": "http://www.w3.org/2004/02/skos/core#exactMatch"
                  },
                  "s": {
                    "type": "uri",
                    "value": "http://www.wikidata.org/entity/Q190875"
                  }
                },
                "response_file": "identifier-mappings-0000.json",
                "response_sha256": "9367516ae65c1d5c9c31a7c16136df237648b3696247e0adfc8f673cd3e0c109"
              }
            ]
          }
        }
      ],
      "values": {
        "concept": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/cortisol",
        "external": "http://purl.obolibrary.org/obo/CHEBI_17650",
        "externalLabel": "cortisol",
        "label": "Cortisol",
        "page": "https://purl.org/okn/frink/kg/cognitive-kb/page/biology/cortisol",
        "source": "https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/cortisol.md",
        "wikidata": "http://www.wikidata.org/entity/Q190875"
      }
    },
    {
      "row_indices": [
        1
      ],
      "row_sha256": "d44737676fa60cafd1990e1b15c410bffdd2f42f3b4e1b020f6dff634b3d4f0f",
      "solutions": [
        {
          "bindings": {
            "concept": {
              "type": "uri",
              "value": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/symbiosis"
            },
            "external": {
              "type": "uri",
              "value": "http://purl.obolibrary.org/obo/GO_0044403"
            },
            "externalLabel": {
              "type": "literal",
              "value": "biological process involved in symbiotic interaction"
            },
            "label": {
              "type": "literal",
              "value": "Symbiosis",
              "xml:lang": "en"
            },
            "page": {
              "type": "uri",
              "value": "https://purl.org/okn/frink/kg/cognitive-kb/page/biology/symbiosis"
            },
            "source": {
              "type": "uri",
              "value": "https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/symbiosis.md"
            },
            "wikidata": {
              "type": "uri",
              "value": "http://www.wikidata.org/entity/Q121610"
            }
          },
          "local": {
            "assertions": [
              {
                "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                "o": {
                  "type": "literal",
                  "value": "Symbiosis",
                  "xml:lang": "en"
                },
                "p": {
                  "type": "uri",
                  "value": "http://www.w3.org/2004/02/skos/core#prefLabel"
                },
                "s": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/symbiosis"
                }
              },
              {
                "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                "o": {
                  "type": "uri",
                  "value": "http://www.w3.org/2004/02/skos/core#Concept"
                },
                "p": {
                  "type": "uri",
                  "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                },
                "s": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/symbiosis"
                }
              },
              {
                "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                "o": {
                  "type": "uri",
                  "value": "https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/symbiosis.md"
                },
                "p": {
                  "type": "uri",
                  "value": "http://purl.org/dc/terms/source"
                },
                "s": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/page/biology/symbiosis"
                }
              },
              {
                "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                "o": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/symbiosis"
                },
                "p": {
                  "type": "uri",
                  "value": "http://purl.org/dc/terms/subject"
                },
                "s": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/page/biology/symbiosis"
                }
              },
              {
                "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                "o": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/symbiosis"
                },
                "p": {
                  "type": "uri",
                  "value": "https://schema.org/about"
                },
                "s": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/page/biology/symbiosis"
                }
              },
              {
                "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                "o": {
                  "type": "uri",
                  "value": "https://schema.org/CreativeWork"
                },
                "p": {
                  "type": "uri",
                  "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                },
                "s": {
                  "type": "uri",
                  "value": "https://purl.org/okn/frink/kg/cognitive-kb/page/biology/symbiosis"
                }
              }
            ],
            "mappings": [
              {
                "assertion": {
                  "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                  "o": {
                    "type": "uri",
                    "value": "http://www.wikidata.org/entity/Q121610"
                  },
                  "p": {
                    "type": "uri",
                    "value": "http://www.w3.org/2004/02/skos/core#exactMatch"
                  },
                  "s": {
                    "type": "uri",
                    "value": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/symbiosis"
                  }
                },
                "reviews": [
                  {
                    "assertions": [
                      {
                        "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                        "o": {
                          "datatype": "http://www.w3.org/2001/XMLSchema#date",
                          "type": "literal",
                          "value": "2026-10-20"
                        },
                        "p": {
                          "type": "uri",
                          "value": "http://purl.org/dc/terms/date"
                        },
                        "s": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ff8cbfa0c0093bd310a3e55ea2c6e8ad84c42cc840d213087d6496a14cc0dffc"
                        }
                      },
                      {
                        "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                        "o": {
                          "type": "literal",
                          "value": "The source defines symbiosis broadly as a close prolonged association between unlike organisms, including beneficial, neutral and harmful cases. The target defines close long-term biological interaction between different organisms. Both denote the association, not mutualism alone or the gene-product processes that enable it.",
                          "xml:lang": "en"
                        },
                        "p": {
                          "type": "uri",
                          "value": "http://purl.org/dc/terms/description"
                        },
                        "s": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ff8cbfa0c0093bd310a3e55ea2c6e8ad84c42cc840d213087d6496a14cc0dffc"
                        }
                      },
                      {
                        "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                        "o": {
                          "type": "uri",
                          "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement"
                        },
                        "p": {
                          "type": "uri",
                          "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                        },
                        "s": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ff8cbfa0c0093bd310a3e55ea2c6e8ad84c42cc840d213087d6496a14cc0dffc"
                        }
                      },
                      {
                        "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                        "o": {
                          "type": "uri",
                          "value": "http://www.w3.org/2004/02/skos/core#exactMatch"
                        },
                        "p": {
                          "type": "uri",
                          "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"
                        },
                        "s": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ff8cbfa0c0093bd310a3e55ea2c6e8ad84c42cc840d213087d6496a14cc0dffc"
                        }
                      },
                      {
                        "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                        "o": {
                          "type": "uri",
                          "value": "http://www.wikidata.org/entity/Q121610"
                        },
                        "p": {
                          "type": "uri",
                          "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#object"
                        },
                        "s": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ff8cbfa0c0093bd310a3e55ea2c6e8ad84c42cc840d213087d6496a14cc0dffc"
                        }
                      },
                      {
                        "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                        "o": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/symbiosis"
                        },
                        "p": {
                          "type": "uri",
                          "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"
                        },
                        "s": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ff8cbfa0c0093bd310a3e55ea2c6e8ad84c42cc840d213087d6496a14cc0dffc"
                        }
                      },
                      {
                        "graph": "https://purl.org/okn/frink/kg/cognitive-kb",
                        "o": {
                          "type": "uri",
                          "value": "https://www.wikidata.org/wiki/Q121610"
                        },
                        "p": {
                          "type": "uri",
                          "value": "http://purl.org/dc/terms/source"
                        },
                        "s": {
                          "type": "uri",
                          "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ff8cbfa0c0093bd310a3e55ea2c6e8ad84c42cc840d213087d6496a14cc0dffc"
                        }
                      }
                    ],
                    "statement": {
                      "type": "uri",
                      "value": "https://purl.org/okn/frink/kg/cognitive-kb/mapping/ff8cbfa0c0093bd310a3e55ea2c6e8ad84c42cc840d213087d6496a14cc0dffc"
                    }
                  }
                ]
              }
            ],
            "origin": {
              "local_sha256": "eb49ccbb6f729f61807912170c30f9a258d700c40c1a81f8f341361f6683049e"
            }
          },
          "multiplicity": 1,
          "remote_label": {
            "assertion": {
              "graph": "https://purl.org/okn/frink/kg/ubergraph",
              "o": {
                "type": "literal",
                "value": "biological process involved in symbiotic interaction"
              },
              "p": {
                "type": "uri",
                "value": "http://www.w3.org/2000/01/rdf-schema#label"
              },
              "s": {
                "type": "uri",
                "value": "http://purl.obolibrary.org/obo/GO_0044403"
              }
            },
            "occurrences": [
              {
                "binding_index": 1,
                "capture_index": 1,
                "graph": "https://purl.org/okn/frink/kg/ubergraph",
                "json_pointer": "/results/bindings/1",
                "lexical_or_metadata_changed": false,
                "materialized": {
                  "graph": "https://purl.org/okn/frink/kg/ubergraph",
                  "o": {
                    "type": "literal",
                    "value": "biological process involved in symbiotic interaction"
                  },
                  "p": {
                    "type": "uri",
                    "value": "http://www.w3.org/2000/01/rdf-schema#label"
                  },
                  "s": {
                    "type": "uri",
                    "value": "http://purl.obolibrary.org/obo/GO_0044403"
                  }
                },
                "query_file": "ubergraph-0000.rq",
                "query_sha256": "4152aa8c03db362b2e13d0c61dec7efcbbc1e26d9e391d3a10f10dbe7bb2bc70",
                "raw": {
                  "o": {
                    "type": "literal",
                    "value": "biological process involved in symbiotic interaction"
                  },
                  "p": {
                    "type": "uri",
                    "value": "http://www.w3.org/2000/01/rdf-schema#label"
                  },
                  "s": {
                    "type": "uri",
                    "value": "http://purl.obolibrary.org/obo/GO_0044403"
                  }
                },
                "response_file": "ubergraph-0000.json",
                "response_sha256": "346a9ffffa37a0b05ed7724588f2796a9b99a3f4625d9b06398b5fbbac757289"
              }
            ]
          },
          "remote_mapping": {
            "assertion": {
              "graph": "https://purl.org/okn/frink/kg/identifier-mappings",
              "o": {
                "type": "uri",
                "value": "http://purl.obolibrary.org/obo/GO_0044403"
              },
              "p": {
                "type": "uri",
                "value": "http://www.w3.org/2004/02/skos/core#exactMatch"
              },
              "s": {
                "type": "uri",
                "value": "http://www.wikidata.org/entity/Q121610"
              }
            },
            "occurrences": [
              {
                "binding_index": 0,
                "capture_index": 0,
                "graph": "https://purl.org/okn/frink/kg/identifier-mappings",
                "json_pointer": "/results/bindings/0",
                "lexical_or_metadata_changed": false,
                "materialized": {
                  "graph": "https://purl.org/okn/frink/kg/identifier-mappings",
                  "o": {
                    "type": "uri",
                    "value": "http://purl.obolibrary.org/obo/GO_0044403"
                  },
                  "p": {
                    "type": "uri",
                    "value": "http://www.w3.org/2004/02/skos/core#exactMatch"
                  },
                  "s": {
                    "type": "uri",
                    "value": "http://www.wikidata.org/entity/Q121610"
                  }
                },
                "query_file": "identifier-mappings-0000.rq",
                "query_sha256": "59b5476822c5ffbc27b588aa7fb5bcda53b776853f06d31d66ac9756b4ef11d6",
                "raw": {
                  "o": {
                    "type": "uri",
                    "value": "http://purl.obolibrary.org/obo/GO_0044403"
                  },
                  "p": {
                    "type": "uri",
                    "value": "http://www.w3.org/2004/02/skos/core#exactMatch"
                  },
                  "s": {
                    "type": "uri",
                    "value": "http://www.wikidata.org/entity/Q121610"
                  }
                },
                "response_file": "identifier-mappings-0000.json",
                "response_sha256": "9367516ae65c1d5c9c31a7c16136df237648b3696247e0adfc8f673cd3e0c109"
              }
            ]
          }
        }
      ],
      "values": {
        "concept": "https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/symbiosis",
        "external": "http://purl.obolibrary.org/obo/GO_0044403",
        "externalLabel": "biological process involved in symbiotic interaction",
        "label": "Symbiosis",
        "page": "https://purl.org/okn/frink/kg/cognitive-kb/page/biology/symbiosis",
        "source": "https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/symbiosis.md",
        "wikidata": "http://www.wikidata.org/entity/Q121610"
      }
    }
  ],
  "joined_rows": 2,
  "limits": "Assertion paths add no composed mapping, identity, class equivalence, membership, reasoning, remote review, scientific or human validation. Hashes do not authenticate authorship. Capture time does not establish freshness or review time. Local blank nodes/lexical normalization are unsupported. Duplicate scalar rows share all typed solutions; individual language assignments cannot be recovered.",
  "origin": {
    "capture_manifest_sha256": "1ea95512c36f97a84bb284ee871cc5a4db8ceceb55cb33841fdf2eea14862809",
    "capture_schema_version": 3,
    "captures": [
      {
        "graph": "https://purl.org/okn/frink/kg/identifier-mappings",
        "query_file": "identifier-mappings-0000.rq",
        "query_sha256": "59b5476822c5ffbc27b588aa7fb5bcda53b776853f06d31d66ac9756b4ef11d6",
        "response_file": "identifier-mappings-0000.json",
        "response_sha256": "9367516ae65c1d5c9c31a7c16136df237648b3696247e0adfc8f673cd3e0c109"
      },
      {
        "graph": "https://purl.org/okn/frink/kg/ubergraph",
        "query_file": "ubergraph-0000.rq",
        "query_sha256": "4152aa8c03db362b2e13d0c61dec7efcbbc1e26d9e391d3a10f10dbe7bb2bc70",
        "response_file": "ubergraph-0000.json",
        "response_sha256": "346a9ffffa37a0b05ed7724588f2796a9b99a3f4625d9b06398b5fbbac757289"
      }
    ],
    "endpoint": "https://apps.okn.us/federation/sparql",
    "join_query_sha256": "9433fa8deac49b575bfbdbb8f0635348a429bd09678c28baa3575cc43d32f35e",
    "local_sha256": "eb49ccbb6f729f61807912170c30f9a258d700c40c1a81f8f341361f6683049e",
    "observed_at": "2026-10-23T17:31:45.038105+00:00",
    "result_basis": "captured-and-recomputed",
    "result_sha256": {
      "join-results.json": "8569ac455aa1cc8f1d168a6f68095356948ddcc5db4775efe9e69c1950e63f19",
      "join-results.md": "11081ee5d93a3aaaf9ad7f315314fe3d30c51a9beb99f1133bb925e2648c4512"
    }
  },
  "producer": {
    "rdflib_version": "7.6.0",
    "source_sha256": {
      "daf_okn.analysis.cross_graph": "a4fedfa0afba24061f4f2d14f7c67b07e15309d0b4a5e7b79bda10a4fea7239e",
      "daf_okn.analysis.join_explanation": "371ac6881f9f804b9ce8f3da7f799700edcf8309df1f678ea76390402707647a",
      "daf_okn.core.alignment": "c7b368c649a3f4126e7f506053cb544243c90242ce344d01979ed530264038ed",
      "daf_okn.core.join_evidence": "d6753b694c82c9f228c44dc4a0cd6b32b7f85d53a156f3b819e242d14182082d",
      "daf_okn.core.join_report": "53fada0930a05b1488416c14173d066639d0b86b69c434e3263065718b96599f",
      "daf_okn.core.rdf": "421433811d49866cc4a6884659d3d2e9b442d32c92aa5f2d70cc282a1b13ae14",
      "daf_okn.core.rdf_submission": "ca58a4d8666c07b0ecfd7ed549a9d34542643ebfa6875af12ac410547c5318e8"
    },
    "version": "0.2.0"
  },
  "profile": "cognitive-obo-join-support-1",
  "semantics": "assertion-path",
  "typed_solution_count": 2
}
```
