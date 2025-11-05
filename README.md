# DPKG
Our work, Multi-Hop Question Generation via Dual-Perspective Keyword Guidance, has been accepted to the Findings of ACL 2025.

Thank you for your interest in our work. If you have any questions after reading our paper, please feel free to contact me at dong_i@163.com or 20254027002@stu.suda.edu.cn


If you use our data or code in your work, please kindly cite our work as:

```bibtex
@inproceedings{li-etal-2025-multi-hop,
    title = "Multi-Hop Question Generation via Dual-Perspective Keyword Guidance",
    author = "Li, Maodong  and
      Zhang, Longyin  and
      Kong, Fang",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.526/",
    doi = "10.18653/v1/2025.findings-acl.526",
    pages = "10096--10112",
    ISBN = "979-8-89176-256-5",
    abstract = "Multi-hop question generation (MQG) aims to generate questions that require synthesizing multiple information snippets from documents to derive target answers. The primary challenge lies in effectively pinpointing crucial information snippets related to question-answer (QA) pairs, typically relying on keywords. However, existing works fail to fully utilize the guiding potential of keywords and neglect to differentiate the distinct roles of question-specific and document-specific keywords. To address this, we define dual-perspective keywords{---}question and document keywords{---}and propose a Dual-Perspective Keyword-Guided (DPKG) framework, which seamlessly integrates keywords into the multi-hop question generation process. We argue that question keywords capture the questioner{'}s intent, whereas document keywords reflect the content related to the QA pair. Functionally, question and document keywords work together to pinpoint essential information snippets in the document, with question keywords required to appear in the generated question. The DPKG framework consists of an expanded transformer encoder and two answer-aware transformer decoders for keyword and question generation, respectively. Extensive experiments on HotpotQA demonstrate the effectiveness of our work, showcasing its promising performance and underscoring its significant value in the MQG task."
}
