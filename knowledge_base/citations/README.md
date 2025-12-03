---
title: Citation Management System
type: reference
status: active
created: 2025-01-01
updated: 2025-01-01
tags:
  - citations
  - references
  - bibliography
  - academic
  - research
semantic_relations:
  - type: manages
    links:
      - [[friston_2017]]
      - [[parr_2019]]
      - [[shannon_1948]]
---

# Citation Management System

This directory contains structured citations and references for the cognitive modeling knowledge base. Citations are organized by author-year format and include comprehensive metadata for research traceability and academic integrity.

## 📚 Citation Organization

### File Naming Convention
- **Format**: `author_year.md` (e.g., `friston_2017.md`)
- **Multiple Authors**: Use first author's name (e.g., `friston_2010.md` for Friston et al.)
- **Multiple Papers**: Append descriptor (e.g., `friston_2017_active_inference.md`)

### Citation Metadata Structure

Each citation file contains:

```yaml
---
title: "Full Paper Title"
authors:
  - "Author One"
  - "Author Two"
type: citation
status: verified
created: 2025-01-01
year: 2017
journal: "Journal Name"
doi: "10.0000/xxxxx"
tags:
  - active_inference
  - free_energy
  - neuroscience
semantic_relations:
  - type: cites
    links:
      - [[../cognitive/active_inference]]
      - [[../mathematics/free_energy_principle]]
---
```

## 🔍 Citation Categories

### Core Active Inference Literature
- [[friston_2017]] - Foundational Active Inference review
- [[parr_2019]] - Active Inference tutorial
- [[friston_2010]] - Free energy principle formulation
- [[friston_2009]] - Predictive coding in neuroscience

### Mathematical Foundations
- [[shannon_1948]] - Information theory foundations
- [[cover_1999]] - Elements of information theory
- [[mackay_2003]] - Information theory, inference, and learning
- [[bishop_2006]] - Pattern recognition and machine learning

### Neuroscience and Cognition
- [[dayan_2008]] - Theoretical neuroscience
- [[frith_2007]] - Making up the mind
- [[clark_2013]] - Whatever next? Predictive brains
- [[hodgkin_1952]] - Quantitative description of membrane current

### Systems Theory and Complexity
- [[prigogine_1984]] - Order out of chaos
- [[kauffman_1993]] - Origins of order
- [[holland_1995]] - Hidden order
- [[barabasi_2016]] - Network science

## 📖 Citation Content Structure

### Required Fields
- **Title**: Full paper/article title
- **Authors**: Complete author list with affiliations
- **Year**: Publication year
- **Journal/Conference**: Publication venue
- **DOI/ISBN**: Digital object identifier or ISBN
- **Abstract**: Paper abstract (if available)
- **Key Contributions**: Main findings and implications
- **Related Concepts**: Links to knowledge base concepts

### Optional Fields
- **PDF Link**: Local or external PDF location
- **Code/Data**: Links to implementations or datasets
- **Citations**: Papers that cite this work
- **Reviews**: Critical assessments or reviews
- **Extensions**: Follow-up work or applications

## 🔗 Integration with Knowledge Base

### Semantic Linking
Citations are semantically linked to relevant concepts:

```yaml
semantic_relations:
  - type: foundational_for
    links:
      - [[../cognitive/active_inference]]
  - type: extends
    links:
      - [[../mathematics/predictive_coding]]
  - type: cited_by
    links:
      - [[friston_2019]]
```

### Cross-References
- Citations link to concepts they inform
- Concepts link to citations that define them
- Related citations are interconnected
- Implementation examples reference citations

## 📊 Citation Quality Standards

### Verification Process
- [ ] DOI validation and accessibility
- [ ] Author and title accuracy
- [ ] Abstract and key findings completeness
- [ ] Semantic relation accuracy
- [ ] Link functionality verification

### Completeness Criteria
- [ ] All major references cited in knowledge base
- [ ] Complete metadata for each citation
- [ ] Bidirectional linking with concepts
- [ ] Regular updates for new literature

## 🛠️ Citation Management Tools

### Automated Validation
```bash
# Validate citation links
find knowledge_base -name "*.md" -exec grep -l "citations/" {} \; | \
while read file; do
    echo "Checking citations in: $file"
    grep -o "\[\[citations/[^]]*\]\]" "$file" | \
    while read citation; do
        citation_file="${citation:12:-2}.md"
        if [ ! -f "knowledge_base/citations/$citation_file" ]; then
            echo "Missing citation: $citation_file"
        fi
    done
done
```

### Citation Import
```python
# Automated citation creation
def create_citation(bibtex_entry):
    """Create citation file from BibTeX entry."""
    metadata = parse_bibtex(bibtex_entry)
    filename = generate_filename(metadata)
    content = generate_markdown(metadata)

    with open(f"knowledge_base/citations/{filename}.md", 'w') as f:
        f.write(content)
```

## 📈 Citation Statistics

### Current Coverage
- **Active Inference**: 15+ core papers
- **Free Energy Principle**: 10+ foundational works
- **Predictive Processing**: 8+ key publications
- **Mathematical Methods**: 12+ technical papers
- **Neuroscience**: 6+ review articles

### Coverage Goals
- **Complete Core Literature**: 50+ foundational papers
- **Cross-Domain Integration**: Full bidirectional linking
- **Regular Updates**: Monthly literature reviews
- **Quality Validation**: 100% verified citations

## 🤝 Contributing Citations

### Addition Process
1. **Identify Gap**: Find uncited important work
2. **Gather Metadata**: Collect complete bibliographic information
3. **Create File**: Follow naming and structure conventions
4. **Add Links**: Establish semantic relations to concepts
5. **Validate**: Ensure all links work and metadata complete

### Quality Guidelines
- Include only peer-reviewed, reputable sources
- Focus on foundational and highly-cited works
- Maintain neutrality and accuracy in summaries
- Link to open-access versions when available
- Update citations with new developments

## 🔄 Maintenance and Updates

### Regular Tasks
- **Monthly Review**: Check for new important publications
- **Link Validation**: Verify all citation links functional
- **Metadata Updates**: Add new semantic relations as needed
- **Quality Audit**: Annual review of citation completeness

### Version Control
- Citation files under version control
- Changes logged with clear commit messages
- Regular backups of citation database
- Collaborative editing with review process

---

> **Research Integrity**: This citation system ensures academic rigor and enables researchers to trace concepts back to their original sources.

---

> **Knowledge Evolution**: Citations provide the foundation for understanding how the field has developed and where it is heading.

---

> **Open Science**: By maintaining comprehensive citations, we support reproducible research and scientific transparency.
