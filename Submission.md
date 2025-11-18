# Homework 7.1 - Vector RAG with Streamlit  
### Kahlil Wassell

---

## 1. Documents Used

Documents included...

- **PDF**
  - **Advanced-1-Marathon-Printable.pdf** - structured marathon training plan with weekly mileage progression and long run scheduling.

- **Markdown**
  - **vo2_max.md** - deep dive into VO2 max, covering physiology, limiting factors, measurement protocols, and training methods.
  - **lactate_threshold.md** - detailed explanation of lactate threshold, metabolic contributors, testing approaches, and strategies to raise LT.

- **Text**
  - **hyrox_endurance_vs_running_endurance.txt** - analysis of the differences between Hyrox endurance demands and traditional distance running.
  - **running_shoe_biomechanics.txt** - overview of midsole foams, carbon plates, stack height, stability design, and injury considerations.
  - **1984.txt** - included as a “wildcard” long form text to test retrieval on non running content; it provides thematic material related to surveillance, authoritarianism, power dynamics, and high level dystopian concepts.

---

## 2. Example Questions and System Behavior

### VO2 Max
- **Question:** “What factors determine VO2 max and how can training improve it?”
- **Behavior:** Retrieved key sections from vo2_max.md covering cardiac output, and other relevant topics.
- **Quality:** The answer stayed grounded in the provided material, avoided hallucinations, and tied physiological factors directly to training methods.

### Lactate Threshold
- **Question:** “How is lactate threshold different from VO2 max for distance runners?”
- **Behavior:** Drew from both lactate_threshold.md and vo2_max.md to produce a combined, coherent explanation.
- **Quality:** Correctly contrasted VO2 max as an aerobic ceiling versus LT as efficiency near that ceiling. The answer remained consistent with the documents.

### Hyrox
- **Question:** “Why does Hyrox feel harder than a regular 10K race?”
- **Behavior:** Retrieved from hyrox_endurance_vs_running_endurance.txt.
- **Quality:** Highlighted muscular fatigue, high lactate tolerance, station based strength demands (sled push/pull, carries, wall balls), and the hybrid nature of the event.

### Running Shoe Biomechanics
- **Question:** “How do carbon plates and high stack foams change running biomechanics?”
- **Behavior:** Retrieved chunks from running_shoe_biomechanics.txt.
- **Quality:** Addressed energy return, forefoot stiffness, muscle loading, and trade offs in stability. The answer closely followed the text.

### Out of Scope Question
- **Question:** “What is the formula for gravitational acceleration?”
- **Behavior:** "The formula for gravitational acceleration is not explicitly mentioned in the provided context.”
- **Quality:** Ideal behavior. The model correctly recognized the question was outside the texts rather than hallucinating.

---

## 3. Critique of Results

Even though the system worked well overall, a few limitations stood out.

### Chunking Artifacts
Some explanations spanned multiple paragraphs but got split across chunk boundaries. As a result, the model occasionally answered from an incomplete conceptual slice.

### No Conversational Memory
Each search is processed independently.Follow up questions like “How does that affect marathon pacing?” aren’t contextually linked to previous answers.

### Context Interpretation Quirks
Because the texts mixes technical running documents with the full text of 1984, the system sometimes interprets questions in unexpected ways. When a search includes words that appear in the novel (like “stars,” “astronomy,” or “planets”), the similarity search may surface literary passages instead of scientific explanations. This leads to answers that blend accurate information with unrelated narrative context. It’s not hallucination, but rather a side effect of embedding based retrieval in a small, mixed domain dataset where semantic overlap can be misleading.

---

## 4. Possible Improvements

If I were to extend this system, I would explore...

- **Multi turn conversational memory**  
  Persist the conversation history in the prompt to support more natural followup questions.

- **Tune chunk size and overlap**  
  Adjust chunking settings to reduce the chance of slicing apart cohesive explanations. Larger overlaps may improve responses.

---

