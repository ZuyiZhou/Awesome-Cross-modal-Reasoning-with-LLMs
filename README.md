# <p align=center> From Linguistic Giants to Sensory Maestros: A Survey on Cross-Modal Reasoning with Large Language Models



#Coming soon！

## <span id="head-content"> *Content* </span>
* - [x] [1. Taxonomy](#head1)

* - [x] [2. Methods for LLMs as Multimodal Fusion Engine](#head2)
  * - [x] [2.1 Methods of Prompt Tuning](#head2-1)
  * - [x] [2.2 Methods of Instruction Tuning](#head2-2)  
  * - [x] [2.3 Methods of Multimodal Pre-training](#head2-3)
  
* - [x] [3. Methods for LLMs as Textual Processor](#head3)
  * - [x] [2.1 Methods of Semantic Refiner](#head3-1)
  * - [x] [2.2 Methods of Content Amplifier](#head3-2)  

* - [x] [4. Methods for LLMs as Multimodal Fusion Engine](#head4)
  * - [x] [2.1 Methods of Programmatic Construction](#head4-1)
  * - [x] [2.2 Methods of Linguistic Interaction](#head4-2)  

* - [x] [5. Methods for LLMs as Multimodal Fusion Engine](#head5)
  * - [x] [2.1 Methods of Prompt Tuning](#head5-1)
  * - [x] [2.2 Methods of Instruction Tuning](#head5-2)  


* [*Contact Us*](#head6)

# <span id="head1"> Taxonomy


# Awesome Papers

## <span id="head2"> Multimodal Fusion Engine

### <span id="head2-1"> Prompt Tuning


|  Model  |   Paper  |   Code   |   Demo   |   Year   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|  CAT  ![Star](https://img.shields.io/github/stars/ttengwang/Caption-Anything.svg?style=social&label=Star) | <br> [**Caption Anything: Interactive Image Description with Diverse Multimodal Controls**](https://arxiv.org/pdf/2305.02677) <br>  | [Github](https://github.com/ttengwang/Caption-Anything) | [Demo](https://huggingface.co/spaces/TencentARC/Caption-Anything) |2023|
|  KAT ![Star](https://img.shields.io/github/stars/guilk/KAT.svg?style=social&label=Star)  | <br> [**KAT: A Knowledge Augmented Transformer for Vision-and-Language**](https://arxiv.org/abs/2112.08614) <br>  | [Github](https://github.com/guilk/KAT) | -- |2022|
| REVIVE  ![Star](https://img.shields.io/github/stars/yuanze-lin/REVIVE.svg?style=social&label=Star)  | <br> [**REVIVE: Regional Visual Representation Matters in Knowledge-Based Visual Question Answering**](https://arxiv.org/abs/2206.01201) <br>  | [Github](https://github.com/yuanze-lin/REVIVE) | -- |2022|
|  Visual ChatGPT ![Star](https://img.shields.io/github/stars/microsoft/TaskMatrix.svg?style=social&label=Star)  | <br>[**Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models**](https://arxiv.org/pdf/2303.04671.pdf)<br>  | [Github](https://github.com/microsoft/TaskMatrix) | [Demo](https://huggingface.co/spaces/microsoft/visual_chatgpt) |2023|
|  PaLM-E ![Star](https://img.shields.io/github/stars/kyegomez/PALM-E.svg?style=social&label=Star) | <br> [**PaLM-E: An Embodied Multimodal Language Model**](https://arxiv.org/pdf/2303.03378.pdf) <br>  | [Github](https://github.com/kyegomez/PALM-E) | [Demo](https://palm-e.github.io/#demo) |2022|
|  VL-T5  ![Star](https://img.shields.io/github/stars/j-min/VL-T5.svg?style=social&label=Star) | <br> [**Unifying Vision-and-Language Tasks via Text Generation**](https://arxiv.org/abs/2102.02779) <br>  | [Github](https://github.com/j-min/VL-T5) | [Demo](https://replicate.com/j-min/vl-t5) |2021|
| BLIP-2   ![Star](https://img.shields.io/github/stars/salesforce/LAVIS.svg?style=social&label=Star) | <br> [**BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**](https://arxiv.org/pdf/2301.12597.pdf) <br>  | [Github](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | [Demo](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/examples/blip2_instructed_generation.ipynb) |2023|
| VPGTrans ![Star](https://img.shields.io/github/stars/VPGTrans/VPGTrans.svg?style=social&label=Star)  | <br> [**VPGTrans: Transfer Visual Prompt Generator across LLMs**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/407106f4b56040b2e8dcad75a6e461e5-Abstract-Conference.html) <br>  | [Github](https://github.com/VPGTrans/VPGTrans) | [Demo](https://ee569fe29733644a33.gradio.live/) |2023|
| eP-ALM ![Star](https://img.shields.io/github/stars/mshukor/eP-ALM.svg?style=social&label=Star)  | <br> [**eP-ALM: Efficient Perceptual Augmentation of Language Models**](https://openaccess.thecvf.com/content/ICCV2023/papers/Shukor_eP-ALM_Efficient_Perceptual_Augmentation_of_Language_Models_ICCV_2023_paper.pdf) <br>  | [Github](https://github.com/mshukor/eP-ALM) | [Demo](https://huggingface.co/mshukor) |2023|
| VCOT | <br> [**Visual Chain of Thought: Bridging Logical Gaps with Multimodal Infillings**](https://arxiv.org/pdf/2305.02317.pdf)<br>  | [Github](https://github.com/dannyrose30/VCOT) | -- |2024|
| VCoder ![Star](https://img.shields.io/github/stars/SHI-Labs/VCoder.svg?style=social&label=Star)  | <br> [**VCoder: Versatile Vision Encoders for Multimodal Large Language Models**](https://openaccess.thecvf.com/content/CVPR2024/html/Jain_VCoder_Versatile_Vision_Encoders_for_Multimodal_Large_Language_Models_CVPR_2024_paper.html) <br>  | [Github](https://github.com/SHI-Labs/VCoder) | [Demo](https://huggingface.co/shi-labs/vcoder_ds_llava-v1.5-13b) |2024|
|  V2L-Tokenizer  ![Star](https://img.shields.io/github/stars/zh460045050/V2L-Tokenizer.svg?style=social&label=Star) | <br> [**Beyond Text: Frozen Large Language Models in Visual Signal Comprehension**](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Beyond_Text_Frozen_Large_Language_Models_in_Visual_Signal_Comprehension_CVPR_2024_paper.pdf) <br>  | [Github](https://github.com/zh460045050/V2L-Tokenizer) | -- |2024|
| BT-Adapter ![Star](https://img.shields.io/github/stars/farewellthree/BT-Adapter.svg?style=social&label=Star)  | <br> [**BT-Adapter: Video Conversation is Feasible Without Video Instruction Tuning**](https://arxiv.org/abs/2309.15785) <br>  | [Github](https://github.com/farewellthree/BT-Adapter) | [Demo](https://huggingface.co/farewellthree/BTAdapter-Weight) |2024|
|  MEAgent | <br> [**Few-Shot Multimodal Explanation for Visual Question Answering**](https://openreview.net/forum?id=jPpK9RzWvh&referrer=%5Bthe%20profile%20of%20Dizhan%20Xue%5D(%2Fprofile%3Fid%3D~Dizhan_Xue1)) <br>  | [Github](https://github.com/LivXue/FS-MEVQA) | -- |2024|

### <span id="head2-2"> Instruction Tuning

|  Model  |   Paper  |   Code   |   Demo   |   Year   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| LION ![Star](https://img.shields.io/github/stars/JiuTian-VL/JiuTian-LION.svg?style=social&label=Star) | <br> [**LION : Empowering Multimodal Large Language Model with Dual-Level Visual Knowledge**](https://arxiv.org/abs/2311.11860) <br>  | [Github](https://github.com/JiuTian-VL/JiuTian-LION) | -- |2024|
| VistaLLM | <br> [**Jack of All Tasks, Master of Many: Designing General-purpose Coarse-to-Fine Vision-Language Model**](https://arxiv.org/abs/2312.12423) <br>  | [Github](https://shramanpramanick.github.io/VistaLLM/) | -- |2024|
|  Chat-UniVi ![Star](https://img.shields.io/github/stars/PKU-YuanGroup/Chat-UniVi.svg?style=social&label=Star) | <br> [**Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding**](https://arxiv.org/abs/2305.13225) <br>  | [Github](https://github.com/PKU-YuanGroup/Chat-UniVi) | [Demo](https://huggingface.co/spaces/Chat-UniVi/Chat-UniVi) |2024|
|  GPT4RoI ![Star](https://img.shields.io/github/stars/jshilong/GPT4RoI.svg?style=social&label=Star) | <br> [**Gpt4roi: Instruction tuning large language model on region-of-interest**](https://arxiv.org/abs/2307.03601) <br>  | [Github](https://github.com/jshilong/GPT4RoI) | [Demo](http://139.196.83.164:7000/) |2024|
| LLaVA ![Star](https://img.shields.io/github/stars/haotian-liu/LLaVA.svg?style=social&label=Star)  | <br> [**Visual instruction tuning**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html) <br>  | [Github](https://llava-vl.github.io/) | [Demo](https://llava.hliu.cc/) |2023|
| InstructBLIP | <br> [**InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning**](https://arxiv.org/abs/2305.06500) <br>  | [Github](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) | -- |2023|
| Video-ChatGPT  ![Star](https://img.shields.io/github/stars/mbzuai-oryx/Video-ChatGPT.svg?style=social&label=Star)  | <br> [**Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models**](https://arxiv.org/abs/2306.05424) <br>  | [Github](https://github.com/mbzuai-oryx/Video-ChatGPT) | [Demo](https://www.ival-mbzuai.com/video-chatgpt) |2024|
| LLaVA-Med ![Star](https://img.shields.io/github/stars/microsoft/LLaVA-Med.svg?style=social&label=Star)  | <br> [**LLaVA-Med: Large Language and Vision Assistant for Biomedicine**](https://arxiv.org/abs/2306.00890) <br>  | [Github](https://github.com/microsoft/LLaVA-Med) | [Demo](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) |2023|
| MedVInt ![Star](https://img.shields.io/github/stars/xiaoman-zhang/PMC-VQA.svg?style=social&label=Star)  | <br> [**Pmc-vqa: Visual instruction tuning for medical visual question**](https://arxiv.org/abs/2305.10415) <br>  | [Github](https://github.com/xiaoman-zhang/PMC-VQA) | [Demo](https://huggingface.co/xmcmic/MedVInT-TE/) |2023|
| Gpt4tools ![Star](https://img.shields.io/github/stars/AILab-CVC/GPT4Tools.svg?style=social&label=Star)  | <br> [**Gpt4tools: Teaching large language model to use tools via self-instruction**](https://arxiv.org/abs/2305.18752) <br>  | [Github](https://github.com/AILab-CVC/GPT4Tools) | [Demo](https://huggingface.co/stevengrove/gpt4tools-vicuna-13b-lora) |2023|
|  MiniGPT-5 ![Star](https://img.shields.io/github/stars/eric-ai-lab/MiniGPT-5.svg?style=social&label=Star) | <br> [**MiniGPT-5: Interleaved Vision-and-Language Generation via Generative Vokens**](https://arxiv.org/abs/2310.02239) <br>  | [Github](https://github.com/eric-ai-lab/MiniGPT-5) | -- |2024|
| MiniGPT-4 ![Star](https://img.shields.io/github/stars/Vision-CAIR/MiniGPT-4.svg?style=social&label=Star)  | <br> [**MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models**](https://arxiv.org/abs/2304.10592) <br>  | [Github](https://github.com/Vision-CAIR/MiniGPT-4) | [Demo](https://huggingface.co/Vision-CAIR/MiniGPT-4) |2023|
| MiniGPT-v2 ![Star](https://img.shields.io/github/stars/Vision-CAIR/MiniGPT-4.svg?style=social&label=Star)  | <br> [**MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning**](https://arxiv.org/abs/2310.09478) <br>  | [Github](https://github.com/Vision-CAIR/MiniGPT-4) | [Demo](https://minigpt-v2.github.io/) |2023|
| VideoChat ![Star](https://img.shields.io/github/stars/OpenGVLab/Ask-Anything.svg?style=social&label=Star)  | <br> [**VideoChat: Chat-Centric Video Understanding**](https://arxiv.org/abs/2305.06355) <br>  | [Github](https://github.com/OpenGVLab/Ask-Anything) | [Demo](https://openxlab.org.cn/apps/detail/yinanhe/VideoChat2) |2023|
|  LaVIN ![Star](https://img.shields.io/github/stars/luogen1996/LaVIN.svg?style=social&label=Star) | <br> [**Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models**](https://arxiv.org/abs/2305.15023)) <br>  | [Github](https://github.com/luogen1996/LaVIN) | -- |2023|
|  Video-LLaMA ![Star](https://img.shields.io/github/stars/DAMO-NLP-SG/Video-LLaMA.svg?style=social&label=Star) | <br> [**Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**](https://arxiv.org/abs/2306.02858) <br>  | [Github](https://github.com/DAMO-NLP-SG/Video-LLaMA) | [Demo](https://huggingface.co/spaces/DAMO-NLP-SG/Video-LLaMA) |2023|
|  DetGPT ![Star](https://img.shields.io/github/stars/OptimalScale/DetGPT.svg?style=social&label=Star) | <br> [**DetGPT: Detect What You Need via Reasoning**](https://arxiv.org/abs/2305.14167) <br>  | [Github](https://github.com/OptimalScale/DetGPT) | [Demo](https://a03e18d54fcb7ceb54.gradio.live/) |2023|
|  Macaw-LLM ![Star](https://img.shields.io/github/stars/lyuchenyang/Macaw-LLM.svg?style=social&label=Star) | <br> [**Macaw-LLM: Multi-Modal Language Modeling with Image, Audio, Video, and Text Integration**](https://arxiv.org/abs/2306.09093) <br>  | [Github](https://github.com/lyuchenyang/Macaw-LLM) | [Demo](https://www.dropbox.com/scl/fo/4ded7qj8my90fes1yxqqd/h?dl=0&rlkey=is2zkfrw76yiidwolgm47x9tv) |2023|
|  LLaMA-Adapter ![Star](https://img.shields.io/github/stars/OpenGVLab/LLaMA-Adapter.svg?style=social&label=Star) | <br> [**LLaMA-Adapter: Efficient Fine-tuning of LLaMA**](https://arxiv.org/abs/2303.16199) <br>  | [Github](https://github.com/OpenGVLab/LLaMA-Adapter) | [Demo](http://imagebind-llm.opengvlab.com/) |2023|
|  LLaMA-Adapter V2 ![Star](https://img.shields.io/github/stars/Alpha-VLLM/LLaMA2-Accessory.svg?style=social&label=Star) | <br> [**LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model**](https://arxiv.org/abs/2304.15010) <br>  | [Github](https://github.com/Alpha-VLLM/LLaMA2-Accessory) | [Demo](http://imagebind-llm.opengvlab.com/) |2023|
| OFA(using MultiInstruct) ![Star](https://img.shields.io/github/stars/VT-NLP/MultiInstruct.svg?style=social&label=Star)  | <br> [**MULTIINSTRUCT: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning**](https://arxiv.org/abs/2205.05638) <br>  | [Github](https://github.com/VT-NLP/MultiInstruct) | -- |2023|
| ChatBridge ![Star](https://img.shields.io/github/stars/joez17/ChatBridge.svg?style=social&label=Star)  | <br> [**ChatBridge: Bridging Modalities with Large Language Model as a Language Catalyst**](https://arxiv.org/pdf/2305.16103.pdf) <br>  | [Github](https://github.com/joez17/ChatBridge) | -- |2023|
| PandaGPT ![Star](https://img.shields.io/github/stars/yxuansu/PandaGPT.svg?style=social&label=Star)  | <br> [**PandaGPT: One Model To Instruction-Follow Them All**](https://arxiv.org/abs/2305.16355) <br>  | [Github](https://github.com/yxuansu/PandaGPT) | [Demo](https://huggingface.co/spaces/GMFTBY/PandaGPT) |2023|
| MultiModal-GPT ![Star](https://img.shields.io/github/stars/open-mmlab/Multimodal-GPT.svg?style=social&label=Star)  | <br> [**MultiModal-GPT: A Vision and Language Model for Dialogue with Humans**](https://arxiv.org/abs/2305.04790v3) <br>  | [Github](https://github.com/open-mmlab/Multimodal-GPT) | -- |2023|
| Ying-VLM ![Star](https://img.shields.io/github/stars/M3-IT/YING-VLM.svg?style=social&label=Star)  | <br> [**M3IT: A Large-Scale Dataset towards Multi-Modal Multilingual Instruction Tuning !**](https://arxiv.org/abs/2306.04387) <br>  | [Github](https://github.com/M3-IT/YING-VLM) | [Demo](https://huggingface.co/MMInstruction/YingVLM) |2023|
| Polite Flamingo ![Star](https://img.shields.io/github/stars/ChenDelong1999/polite-flamingo.svg?style=social&label=Star)  | <br> [**Visual Instruction Tuning with Polite Flamingo**](https://arxiv.org/abs/2307.01003) <br>  | [Github](https://github.com/ChenDelong1999/polite-flamingo) | [Demo](https://github.com/ChenDelong1999/polite-flamingo) |2023|
| ChatSpot ![Star](https://img.shields.io/github/stars/Ahnsun/ChatSpot.svg?style=social&label=Star)  | <br> [**ChatSpot: Bootstrapping Multimodal LLMs via Precise Referring Instruction Tuning**](https://arxiv.org/abs/2307.09474) <br>  | [Github](https://github.com/Ahnsun/ChatSpot) | [Demo](https://chatspot.streamlit.app/) |2023|
| BLIVA ![Star](https://img.shields.io/github/stars/mlpc-ucsd/BLIVA.svg?style=social&label=Star)  | <br> [**BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual**](https://arxiv.org/pdf/2308.09936) <br>  | [Github](https://github.com/mlpc-ucsd/BLIVA) | [Demo](https://github.com/mlpc-ucsd/BLIVA?tab=readme-ov-file) |2024|
| BuboGPT ![Star](https://img.shields.io/github/stars/magic-research/bubogpt.svg?style=social&label=Star)  | <br> [**BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs**](https://arxiv.org/pdf/2307.08581.pdf) <br>  | [Github](https://github.com/magic-research/bubogpt) | [Demo](https://huggingface.co/datasets/magicr/BuboGPT) |2023|
| VisionLLM ![Star](https://img.shields.io/github/stars/OpenGVLab/VisionLLM.svg?style=social&label=Star)  | <br> [**VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks**](https://arxiv.org/pdf/2305.11175.pdf) <br>  | [Github](https://github.com/OpenGVLab/VisionLLM) | -- |2023|
| LAMM ![Star](https://img.shields.io/github/stars/OpenLAMM/LAMM.svg?style=social&label=Star)  | <br> [**LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark**](https://arxiv.org/pdf/2306.06687.pdf) <br>  | [Github](https://github.com/OpenLAMM/LAMM) | [Demo](https://huggingface.co/spaces/openlamm/LAMM) |2023|
| Qwen-VL ![Star](https://img.shields.io/github/stars/QwenLM/Qwen-VL.svg?style=social&label=Star)  | <br>[**Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities**](https://arxiv.org/pdf/2308.12966.pdf) <br>  | [Github](https://github.com/QwenLM/Qwen-VL) | [Demo](https://modelscope.cn/studios/qwen/Qwen-VL-Chat-Demo/summary) |2023|
| mPLUG-Owl3 ![Star](https://img.shields.io/github/stars/X-PLUG/mPLUG-Owl.svg?style=social&label=Star)  | <br> [**mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models**](https://www.arxiv.org/pdf/2408.04840) <br>  | [Github](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl3) | -- |2024|
| mPLUG-Owl2 ![Star](https://img.shields.io/github/stars/X-PLUG/mPLUG-Owl.svg?style=social&label=Star)  | <br> [**mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration**](https://arxiv.org/pdf/2311.04257.pdf) <br>  | [Github](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2) | [Demo](https://modelscope.cn/studios/damo/mPLUG-Owl2/summary) |2023|
| mPLUG-Owl ![Star](https://img.shields.io/github/stars/X-PLUG/mPLUG-Owl.svg?style=social&label=Star)  | <br> [**mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality**](https://arxiv.org/pdf/2304.14178.pdf)<br>  | [Github](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl) | [Demo](https://huggingface.co/spaces/MAGAer13/mPLUG-Owl) |2023|
| X-LLM ![Star](https://img.shields.io/github/stars/phellonchen/X-LLM.svg?style=social&label=Star)  | <br> [**X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages**](https://arxiv.org/abs/2305.04160) <br>  | [Github](https://github.com/phellonchen/X-LLM) | -- |2023|
| NExT-GPT ![Star](https://img.shields.io/github/stars/NExT-GPT/NExT-GPT.svg?style=social&label=Star)  | <br> [**NExT-GPT: Any-to-Any Multimodal LLM**](https://arxiv.org/pdf/2309.05519) <br>  | [Github](https://github.com/NExT-GPT/NExT-GPT) | [Demo](https://acc414b22d6839d28f.gradio.live/) |2023|
| SpeechGPT ![Star](https://img.shields.io/github/stars/0nutation/SpeechGPT.svg?style=social&label=Star)  | <br> [**SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities**](https://arxiv.org/abs/2305.11000) <br>  | [Github](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt) | [Demo](https://huggingface.co/fnlp/SpeechGPT-7B-cm) |2023|

### <span id="head2-3"> Multimodal Pre-training

|  Model  |   Paper  |   Code   |   Demo   |   Year   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/facebookresearch/LaViLa.svg?style=social&label=Star)  | <br> [**Learning Video Representations from Large Language Models**](https://arxiv.org/abs/2212.04501) <br>  | [Github](https://github.com/facebookresearch/LaViLa) | [Demo](https://huggingface.co/spaces/nateraw/lavila) |2023|
| ![Star](https://img.shields.io/github/stars/mlfoundations/open_flamingo.svg?style=social&label=Star)  | <br> [**Flamingo: a Visual Language Model for Few-Shot Learning**](https://arxiv.org/abs/2204.14198) <br>  | [Github](https://github.com/mlfoundations/open_flamingo) | [Demo](https://huggingface.co/spaces/dhansmair/flamingo-mini-cap) |2022|
| VideoCoCa | <br> [**VideoCoCa: Video-Text Modeling with Zero-Shot Transfer from Contrastive Captioners**](https://arxiv.org/abs/2212.04979) <br>  | --| -- |2023|
| OpenFlamingo ![Star](https://img.shields.io/github/stars/openflamingo/OpenFlamingo.svg?style=social&label=Star)  | <br> [**OpenFlamingo: An Open-Source Framework for Training Large Autoregressive Vision-Language Models**](https://arxiv.org/abs/2308.01390) <br>  | [Github](https://github.com/mlfoundations/open_flamingo) | [Demo](https://huggingface.co/spaces/openflamingo/OpenFlamingo) |2023|
| Otter ![Star](https://img.shields.io/github/stars/Luodian/Otter.svg?style=social&label=Star)  | <br> [**MIMIC-IT: Multi-Modal In-Context Instruction Tuning**](https://arxiv.org/abs/2306.05425) <br>  | [Github](https://github.com/Luodian/Otter) | [Demo](https://openxlab.org.cn/models/detail/YuanhanZhang/OTTER-Image-MPT7B) |2023|
| Cheetor  ![Star](https://img.shields.io/github/stars/DCDmllm/Cheetah.svg?style=social&label=Star)  | <br> [**Fine-tuning Multimodal LLMs to Follow Zero-shot Demonstrative Instructions**](https://arxiv.org/abs/2308.04152) <br>  | [Github](https://github.com/DCDmllm/Cheetah) | -- |2024|
| MAGMA ![Star](https://img.shields.io/github/stars/Aleph-Alpha/magma.svg?style=social&label=Star)  | <br> [**MAGMA -- Multimodal Augmentation of Generative Models through Adapter-based Finetuning**](https://arxiv.org/abs/2112.05253) <br>  | [Github](https://github.com/Aleph-Alpha/magma) | -- |2022|
| Prismer ![Star](https://img.shields.io/github/stars/NVlabs/prismer.svg?style=social&label=Star)  | <br> [**Prismer: A Vision-Language Model with Multi-Task Experts**](https://arxiv.org/abs/2303.02506) <br>  | [Github](https://github.com/NVlabs/prismer) | [Demo](https://huggingface.co/spaces/lorenmt/prismer) |2024|
| PaLI  ![Star](https://img.shields.io/github/stars/kyegomez/PALI.svg?style=social&label=Star)  | <br> [**PaLI: A Jointly-Scaled Multilingual Language-Image Model**](https://arxiv.org/abs/2209.06794) <br>  | [Github](https://github.com/kyegomez/PALI) | -- |2022|
| LAVILA ![Star](https://img.shields.io/github/stars/facebookresearch/LaViLa.svg?style=social&label=Star)  | <br> [**Learning Video Representations from Large Language Models**](https://arxiv.org/abs/2212.04501) <br>  | [Github](https://github.com/facebookresearch/LaViLa) | [Demo](https://arxiv.org/abs/2212.04501) |2023|
| CoCa ![Star](https://img.shields.io/github/stars/lucidrains/CoCa-pytorch.svg?style=social&label=Star)  | <br> [**CoCa: Contrastive Captioners are Image-Text Foundation Models**](https://arxiv.org/abs/2205.01917) <br>  | [Github](https://github.com/lucidrains/CoCa-pytorch) | -- |2022|
| Gato ![Star](https://img.shields.io/github/stars/OrigamiDream/gato.svg?style=social&label=Star)  | <br> [**A Generalist Agent**](https://arxiv.org/abs/2205.06175) <br>  | [Github](https://github.com/OrigamiDream/gato) | -- |2022|
| PaLI-X | <br> [**PaLI-X: On Scaling up a Multilingual Vision and Language Model**](https://arxiv.org/abs/2305.18565) <br>  | -- | -- |2023|
| COSA ![Star](https://img.shields.io/github/stars/TXH-mercury/COSA.svg?style=social&label=Star)  | <br> [**COSA: Concatenated Sample Pretrained Vision-Language Foundation Model**](https://arxiv.org/abs/2306.09085) <br>  | [Github](https://github.com/TXH-mercury/COSA) | [Demo](https://drive.google.com/file/d/1jaKFGbVE-BW3x5JUjRHbRqhVaXIy8q8s/view) |2023|
| GIT ![Star](https://img.shields.io/github/stars/microsoft/GenerativeImage2Text.svg?style=social&label=Star)  | <br> [**GIT: A Generative Image-to-text Transformer for Vision and Language**](https://arxiv.org/abs/2205.14100) <br>  | [Github](https://github.com/microsoft/GenerativeImage2Text) | -- |2023|
| BEiT-3  | <br> [**Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks**](https://arxiv.org/abs/2208.10442) <br>  | [Github](https://github.com/microsoft/unilm/tree/master/beit3) | -- |2022|
| VLMo  | <br> [**Vlmo: Unified vision-language pretraining with mixture-of-modality-experts**](https://arxiv.org/abs/2111.02358) <br>  | [Github](https://github.com/microsoft/unilm/tree/master/vlmo) | [Demo](https://github.com/wenhui0924/vlmo_ckpts/releases/download/vlmo/vlmo_base_patch16_224.pt) |2022|
| Kosmos-1 | <br> [**Language Is Not All You Need: Aligning Perception with Language Models**](https://arxiv.org/abs/2302.14045) <br>  | [Github](https://github.com/bjoernpl/KOSMOS_reimplementation) | -- |2022|
| Kosmos-2 | <br> [**(Kosmos-2: Grounding Multimodal Large Language Models to the World**](https://arxiv.org/abs/2306.14824) <br>  | [Github](https://github.com/microsoft/unilm/tree/master/kosmos-2) | [Demo](https://huggingface.co/spaces/ydshieh/Kosmos-2) |2023|
| Unified-IO ![Star](https://img.shields.io/github/stars/allenai/unified-io-inference.svg?style=social&label=Star)  | <br> [**Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks**](https://arxiv.org/abs/2206.08916) <br>  | [Github](https://github.com/allenai/unified-io-inference) | -- |2023|
| mPLUG-2 ![Star](https://img.shields.io/github/stars/X-PLUG/mPLUG-2.svg?style=social&label=Star)  | <br> [**mPLUG-2: A Modularized Multi-modal Foundation Model Across Text, Image and Video**](https://arxiv.org/abs/2302.00402) <br>  | [Github](https://github.com/X-PLUG/mPLUG-2) | -- |2023|
| MetaLM  | <br> [**Language Models are General-Purpose Interfaces**](https://arxiv.org/abs/2206.06336) <br>  | [Github](https://github.com/microsoft/unilm/tree/master/metalm) | -- |2022|



## <span id="head3"> Textual Processor

### <span id="head3-1"> Semantic Refiner


|  Model  |   Paper  |   Code   |   Demo   |   Year   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| Socratic Models ![Star](https://img.shields.io/github/stars/abhinav-neil/socratic-models.svg?style=social&label=Star)  | <br> [**Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language**](https://arxiv.org/abs/2204.00598) <br>  | [Github](https://github.com/abhinav-neil/socratic-models) | -- |2022|
| PromptCap ![Star](https://img.shields.io/github/stars/Yushi-Hu/PromptCap.svg?style=social&label=Star)  | <br> [**PromptCap: Prompt-Guided Task-Aware Image Captioning**](https://arxiv.org/abs/2211.09699) <br>  | [Github](https://github.com/Yushi-Hu/PromptCap) | [Demo](https://huggingface.co/tifa-benchmark/promptcap-coco-vqa) |2023|
| img2llm  | <br> [**From Images to Textual Prompts: Zero-shot VQA with Frozen Large Language Models**](https://arxiv.org/abs/2212.10846) <br>  | [Github](https://github.com/salesforce/LAVIS/tree/main/projects/img2llm-vqa) |-- |2023|
| MoqaGPT ![Star](https://img.shields.io/github/stars/lezhang7/MOQAGPT.svg?style=social&label=Star)  | <br> [**MoqaGPT : Zero-Shot Multi-modal Open-domain Question Answering with Large Language Model**](https://arxiv.org/abs/2310.13265) <br>  | [Github](https://github.com/lezhang7/MOQAGPT) | -- |2023|
| VAST ![Star](https://img.shields.io/github/stars/TXH-mercury/VAST.svg?style=social&label=Star)  | <br> [**VAST: A Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset**](https://arxiv.org/abs/2305.18500) <br>  | [Github](https://github.com/TXH-mercury/VAST) | -- |2023|

### <span id="head3-2"> Content Amplifier

|  Model  |   Paper  |   Code   |   Demo   |   Year   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| PointCLIP V2 ![Star](https://img.shields.io/github/stars/yangyangyang127/PointCLIP_V2.svg?style=social&label=Star)  | <br> [**PointCLIP V2: Prompting CLIP and GPT for Powerful 3D Open-world Learning**](https://arxiv.org/abs/2211.11682) <br>  | [Github](https://github.com/yangyangyang127/PointCLIP_V2) | -- |2023|
| ViewRefer ![Star](https://img.shields.io/github/stars/Ivan-Tang-3D/ViewRefer3D.svg?style=social&label=Star)  | <br> [**ViewRefer: Grasp the Multi-view Knowledge for 3D Visual Grounding with GPT and Prototype Guidance**](https://arxiv.org/abs/2303.16894) <br>  | [Github](https://github.com/Ivan-Tang-3D/ViewRefer3D) | -- |2023|
| IdealGPT ![Star](https://img.shields.io/github/stars/Hxyou/IdealGPT.svg?style=social&label=Star)  | <br> [**IdealGPT: Iteratively Decomposing Vision and Language Reasoning via Large Language Models**](https://arxiv.org/pdf/2305.14985.pdf) <br>  | [Github](https://github.com/Hxyou/IdealGPT) | -- |2023|
| UnifiedQA ![Star](https://img.shields.io/github/stars/allenai/unifiedqa.svg?style=social&label=Star)  | <br> [**UnifiedQA: Crossing Format Boundaries With a Single QA System**](https://arxiv.org/abs/2005.00700) <br>  | [Github](https://github.com/allenai/unifiedqa) | [Demo]( https://unifiedqa.apps.allenai.org/) |2020|

## <span id="head4"> Cognitive Controller

### <span id="head4-1"> Programmantic Construction


|  Model  |   Paper  |   Code   |   Demo   |   Year   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/microsoft/JARVIS.svg?style=social&label=Star)  | <br> [**HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**](https://arxiv.org/abs/2303.17580) <br>  | [Github](https://github.com/microsoft/JARVIS/tree/main/hugginggpt) | [Demo](https://huggingface.co/spaces/microsoft/HuggingGPT) |2023|
| Chameleon ![Star](https://img.shields.io/github/stars/lupantech/chameleon-llm.svg?style=social&label=Star)  | <br> [**Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models**](https://arxiv.org/abs/2304.09842) <br>  | [Github](https://github.com/lupantech/chameleon-llm) | -- |2023|
| AssistGPT ![Star](https://img.shields.io/github/stars/showlab/assistgpt.svg?style=social&label=Star)  | <br> [**AssistGPT: A General Multi-modal Assistant that can Plan, Execute, Inspect, and Learn**](https://arxiv.org/abs/2306.08640) <br>  | [Github](https://github.com/showlab/assistgpt) | -- |2023|
| ProgPrompt ![Star](https://img.shields.io/github/stars/NVlabs/progprompt-vh.svg?style=social&label=Star)  | <br> [**ProgPrompt: Generating Situated Robot Task Plans using Large Language Models**](https://arxiv.org/abs/2209.11302) <br>  | [Github](https://github.com/NVlabs/progprompt-vh) | -- |2022|
| Instruct2Act ![Star](https://img.shields.io/github/stars/allenai/visprog.svg?style=social&label=Star)  | <br> [**Visual Programming: Compositional visual reasoning without training**](https://arxiv.org/abs/2211.11559) <br>  | [Github](https://github.com/allenai/visprog) | -- |2023|
| ![Star](https://img.shields.io/github/stars/OpenGVLab/Instruct2Act.svg?style=social&label=Star)  | <br> [**Instruct2Act: Mapping Multi-modality Instructions to Robotic Actions with Large Language Model**](https://arxiv.org/abs/2305.11176) <br>  | [Github](https://github.com/OpenGVLab/Instruct2Act) | [Demo]() |20|
| ViperGPT ![Star](https://img.shields.io/github/stars/cvlab-columbia/viper.svg?style=social&label=Star)  | <br> [**ViperGPT: Visual Inference via Python Execution for Reasoning**](https://arxiv.org/abs/2303.08128) <br>  | [Github](https://github.com/cvlab-columbia/viper) | -- |2023|
| ProViQ  | <br> [**Zero-Shot Video Question Answering with Procedural Programs**](https://arxiv.org/abs/2312.00937) <br>  | -- | --|2023|
|  MEAgent | <br> [**Few-Shot Multimodal Explanation for Visual Question Answering**](https://openreview.net/forum?id=jPpK9RzWvh&referrer=%5Bthe%20profile%20of%20Dizhan%20Xue%5D(%2Fprofile%3Fid%3D~Dizhan_Xue1)) <br>  | [Github](https://github.com/LivXue/FS-MEVQA) | -- |2024|
### <span id="head4-2"> Linguistic Interaction

|  Model  |   Paper  |   Code   |   Demo   |   Year   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| MoqaGPT ![Star](https://img.shields.io/github/stars/lezhang7/MOQAGPT.svg?style=social&label=Star)  | <br> [**MoqaGPT : Zero-Shot Multi-modal Open-domain Question Answering with Large Language Model**](https://arxiv.org/abs/2310.13265) <br>  | [Github](https://github.com/lezhang7/MOQAGPT) | -- |2023|
| ChatCaptioner ![Star](https://img.shields.io/github/stars/Vision-CAIR/ChatCaptioner.svg?style=social&label=Star)  | <br> [**ChatGPT Asks, BLIP-2 Answers: Automatic Questioning Towards Enriched Visual Descriptions**](https://arxiv.org/abs/2303.06594) <br>  | [Github](https://github.com/Vision-CAIR/ChatCaptioner) | -- |2023|
| Video ChatCaptioner ![Star](https://img.shields.io/github/stars/Vision-CAIR/ChatCaptioner.svg?style=social&label=Star)  | <br> [**Video ChatCaptioner: Towards Enriched Spatiotemporal Descriptions**](https://arxiv.org/abs/2304.04227) <br>  | [Github](https://github.com/Vision-CAIR/ChatCaptioner/tree/main/Video_ChatCaptioner) | -- |2023|
| Inner Monologue  | <br> [**Inner Monologue: Embodied Reasoning through Planning with Language Models**](https://arxiv.org/abs/2207.05608) <br>  | [Github](https://innermonologue.github.io/) | -- |2022|
| Shikra ![Star](https://img.shields.io/github/stars/shikras/shikra.svg?style=social&label=Star)  | <br> [**Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic**](https://arxiv.org/abs/2306.15195) <br>  | [Github](https://github.com/shikras/shikra) | [Demo](shikras/shikra) |2023|
| MM-REACT ![Star](https://img.shields.io/github/stars/microsoft/MM-REACT.svg?style=social&label=Star)  | <br> [**MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action**](MM-ReAct) <br>  | [Github](https://github.com/microsoft/MM-REACT) | [Demo](https://huggingface.co/spaces/microsoft-cognitive-service/mm-react) |2023|
| SayCan ![Star](https://img.shields.io/github/stars/kyegomez/SayCan.svg?style=social&label=Star)  | <br> [**Do As I Can, Not As I Say: Grounding Language in Robotic Affordances**](https://arxiv.org/abs/2204.01691) <br>  | [Github](https://github.com/kyegomez/SayCan) | --|2022|
| SpeechGPT ![Star](https://img.shields.io/github/stars/0nutation/SpeechGPT.svg?style=social&label=Star)  | <br> [**SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities**](https://arxiv.org/abs/2305.11000) <br>  | [Github](https://github.com/0nutation/SpeechGPT) | [Demo]-- |2023|
| WALL-E  | <br> [**WALL-E: Embodied Robotic WAiter Load Lifting with Large Language Model**](https://arxiv.org/abs/2308.15962) <br>  | [Github](https://star-uu-wang.github.io/WALL-E/#) | -- |2023|

## <span id="head5"> Knowledge Enhancer

### <span id="head5-1"> Implicit Cognition


|  Model  |   Paper  |   Code   |   Demo   |   Year   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| PICa ![Star](https://img.shields.io/github/stars/microsoft/PICa.svg?style=social&label=Star)  | <br> [**An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA**](https://arxiv.org/abs/2109.05014) <br>  | [Github](https://github.com/microsoft/PICa) | -- |2021|
| ViewRefer ![Star](https://img.shields.io/github/stars/Ivan-Tang-3D/ViewRefer3D.svg?style=social&label=Star)  | <br> [**ViewRefer: Grasp the Multi-view Knowledge for 3D Visual Grounding with GPT and Prototype Guidance**](https://arxiv.org/abs/2303.16894) <br>  | [Github](https://github.com/Ivan-Tang-3D/ViewRefer3D) | -- |2023|
| PromptCap ![Star](https://img.shields.io/github/stars/Yushi-Hu/PromptCap.svg?style=social&label=Star)  | <br> [**PromptCap: Prompt-Guided Task-Aware Image Captioning**](https://arxiv.org/abs/2211.09699) <br>  | [Github](https://github.com/Yushi-Hu/PromptCap) | [Demo](https://huggingface.co/tifa-benchmark/promptcap-coco-vqa) |2023|
|  KAT ![Star](https://img.shields.io/github/stars/guilk/KAT.svg?style=social&label=Star)  | <br> [**KAT: A Knowledge Augmented Transformer for Vision-and-Language**](https://arxiv.org/abs/2112.08614) <br>  | [Github](https://github.com/guilk/KAT) | -- |2022|
| REVIVE  ![Star](https://img.shields.io/github/stars/yuanze-lin/REVIVE.svg?style=social&label=Star)  | <br> [**REVIVE: Regional Visual Representation Matters in Knowledge-Based Visual Question Answering**](https://arxiv.org/abs/2206.01201) <br>  | [Github](https://github.com/yuanze-lin/REVIVE) | -- |2022|
| CaFo ![Star](https://img.shields.io/github/stars/ZrrSkywalker/CaFo.svg?style=social&label=Star)  | <br> [**Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners**](https://arxiv.org/abs/2303.02151) <br>  | [Github](https://github.com/ZrrSkywalker/CaFo) | -- |2023|
| CuPL ![Star](https://img.shields.io/github/stars/sarahpratt/CuPL.svg?style=social&label=Star)  | <br> [**What does a platypus look like? Generating customized prompts for zero-shot image classification**](https://arxiv.org/abs/2209.03320) <br>  | [Github](https://github.com/sarahpratt/CuPL) | -- |2022|
| Prophet ![Star](https://img.shields.io/github/stars/MILVLG/prophet.svg?style=social&label=Star)  | <br> [**Prophet: Prompting Large Language Models with Complementary Answer Heuristics for Knowledge-based Visual Question Answering**](https://arxiv.org/abs/2303.01903) <br>  | [Github](https://github.com/MILVLG/prophet) | -- |2023|
| Proofread ![Star](https://img.shields.io/github/stars/JindongGu/Awesome-Prompting-on-Vision-Language-Model.svg?style=social&label=Star)  | <br> [**Prompting Vision Language Model with Knowledge from Large Language Model for Knowledge-Based VQA**](https://arxiv.org/abs/2308.15851) <br>  | [Github](https://github.com/JindongGu/Awesome-Prompting-on-Vision-Language-Model) | -- |2023|



### <span id="head5-2"> Augmented Knowledge


|  Model  |   Paper  |   Code   |   Demo   |   Year   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| MM-REACT ![Star](https://img.shields.io/github/stars/microsoft/MM-REACT.svg?style=social&label=Star)  | <br> [**MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action**](MM-ReAct) <br>  | [Github](https://github.com/microsoft/MM-REACT) | [Demo](https://huggingface.co/spaces/microsoft-cognitive-service/mm-react) |2023|
| ![Star](https://img.shields.io/github/stars/microsoft/JARVIS.svg?style=social&label=Star)  | <br> [**HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**](https://arxiv.org/abs/2303.17580) <br>  | [Github](https://github.com/microsoft/JARVIS/tree/main/hugginggpt) | [Demo](https://huggingface.co/spaces/microsoft/HuggingGPT) |2023|
| Chameleon ![Star](https://img.shields.io/github/stars/lupantech/chameleon-llm.svg?style=social&label=Star)  | <br> [**Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models**](https://arxiv.org/abs/2304.09842) <br>  | [Github](https://github.com/lupantech/chameleon-llm) | -- |2023|
| LLaVA-Med ![Star](https://img.shields.io/github/stars/microsoft/LLaVA-Med.svg?style=social&label=Star)  | <br> [**LLaVA-Med: Large Language and Vision Assistant for Biomedicine**](https://arxiv.org/abs/2306.00890) <br>  | [Github](https://github.com/microsoft/LLaVA-Med) | [Demo](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) |2023|
| MedVInt ![Star](https://img.shields.io/github/stars/xiaoman-zhang/PMC-VQA.svg?style=social&label=Star)  | <br> [**Pmc-vqa: Visual instruction tuning for medical visual question**](https://arxiv.org/abs/2305.10415) <br>  | [Github](https://github.com/xiaoman-zhang/PMC-VQA) | [Demo](https://huggingface.co/xmcmic/MedVInT-TE/) |2023|




## <span id="head6"> *Contact Us* </span>

Please contact us by e-mail:

```bash
xuedizhan17@mails.ucas.ac.cn
```

```bash
shengsheng.qian@nlpr.ia.ac.cn
```

```bash
zhouzuyi2023@ia.ac.cn
```