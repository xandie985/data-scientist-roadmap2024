**Reducing Hallucinations from RAG**
- Using prompt template: 
  - Final input to the input: RAG data + query_prompt
  - At the end of the query_prompt, mention the llm to provide the answer only if there is contextual information about the query, and should avoid anwering if the context is missing.
- Pass the same query_prompt multiple times, if the responses have different contextual meaning, the LLM is hallucinating.
1. Fact Verification:
   - Cross-Verification: Implement mechanisms to cross-verify the information retrieved from the knowledge base with other reliable sources or external APIs. This can help detect inconsistencies or inaccuracies and provide a more reliable response.
   - Citation and Evidence: Include citations or links to the source documents from which the information was retrieved, allowing users to verify the information themselves and building trust in the system.
2. Confidence Estimation:
   -  Uncertainty Modeling: Train the RAG model to estimate its confidence in each generated answer. This can help identify responses that are more likely to be hallucinations and require further verification.
   -  Thresholding: Set confidence thresholds for different actions. For example, high-confidence answers can be provided directly, while low-confidence answers can be flagged for review or supplemented with additional context.
3. Prompt Engineering:
   - Clear and Specific Prompts: Design prompts that encourage the LLM to focus on factual information and avoid speculation. Avoid open-ended questions or prompts that could lead to creative but inaccurate responses.
   - Constraint Prompts: Include constraints in the prompt that restrict the LLM's output to specific formats or information sources, reducing the likelihood of hallucinations.
4. Robust Retrieval:
   - Hybrid Retrieval: Combine different retrieval methods (e.g., dense and sparse retrieval) to improve the quality and diversity of retrieved documents. This can provide the LLM with a wider range of information to draw from, reducing the reliance on a single source and minimizing the risk of hallucinations.
   - Re-ranking: Apply re-ranking algorithms to prioritize the most relevant and reliable documents, ensuring that the LLM has access to the most accurate information.
5. Feedback and Iteration:
   - Human-in-the-Loop (HITL): Involve human experts to review and correct hallucinated responses, providing valuable feedback to improve the model over time.
   - Active Learning: Identify areas where the model is prone to hallucinations and actively collect more data or fine-tune the model to address those specific weaknesses.
6. Tools and Libraries:
   - LangChain: A popular framework for building RAG systems that offers various components for addressing hallucinations, such as `RetrievalQA` with confidence estimation and source verification.
   - Haystack: An open-source framework for building NLP applications that includes tools for evaluating and mitigating hallucinations in RAG systems.
  - 