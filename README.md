# AIM-XavierEdgeAI
Milvus Lite, Python, NVIDIA Xavier NX, Edge AI, Edge Vector DB



For lower powered See: https://github.com/tspannhw/AIM-RPIAIKit

See:   https://github.com/tspannhw/FLaNK-EdgeAI



### Run insert

SLACK_BOT_TOKEN="xkey" python3 xavier.py

### Search

SLACK_BOT_TOKEN="xkey" python3 xaviersearch.py

### Search and Displau

SLACK_BOT_TOKEN="xkey" python3 xaviersearchdisplay.py

### Milvus Lite Database

````
-rw-r--r-- 1 root root 49152 Jun 27 16:43 XavierEdgeAI.db

````

#### Example Search Run

````
cd /home/nvidia/nvme
python3.11 -m venv milvusvenv
source milvusvenv/bin/activate
cd AIM-XavierEdgeAI
(milvusvenv) root@nvidia-desktop:/home/nvidia/nvme/AIM-XavierEdgeAI# ./search.sh
nvidia-desktop
06/27/2024 16:47:43
/home/nvidia/nvme/images/xavier67d354f0-d8b1-4a97-9fa2-6e35eed85f2d.jpg with caption: there is a blue rubber duck sitting on a computer desk
/home/nvidia/nvme/images/xavierc4738900-22fc-4dc2-8db1-23661eb185f7.jpg with caption: there is a blue toy duck sitting on a computer desk
/home/nvidia/nvme/images/xavierf42f26f6-df89-4510-a3a7-d1888b0b5dc2.jpg with caption: there is a close up of a bunch of wires and wires
/home/nvidia/nvme/images/xavier7fc1a6b5-6289-4d6e-b2be-186c023ce2a0.jpg with caption: there is a close up of a bunch of wires and wires
/home/nvidia/nvme/images/xavier2fb1bfe7-97cb-412f-8599-76860b3bc9ee.jpg with caption: there is a computer and a keyboard on a desk with wires
/home/nvidia/nvme/images/xavierc8a07f8b-36ac-47bf-8e84-338218c4df66.jpg with caption: there is a computer monitor and a keyboard on a desk
/home/nvidia/nvme/images/xavier97553c2d-7035-403c-9953-091dfd213009.jpg with caption: there is a close up of a bunch of wires and wires
/home/nvidia/nvme/images/xavierc7dc6986-ccc4-4d74-929f-590dd098c3c5.jpg with caption: there is a white van with a red stripe on the side
/home/nvidia/nvme/images/xavier5faa7ab4-9671-491d-a387-b6593b9bb7bf.jpg with caption: there is a blue rubber glove on a desk next to a computer
Search done

(milvusvenv) root@nvidia-desktop:/home/nvidia/nvme/AIM-XavierEdgeAI# ./run.sh
nvidia-desktop
06/27/2024 16:49:07
/home/nvidia/nvme/milvusvenv/lib/python3.11/site-packages/transformers/generation/utils.py:1249: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.
  warnings.warn(
/home/nvidia/nvme/images/xavierb10268af-69af-4b38-a254-cca103de76af.jpg
Caption: there is a small toy truck that is on a desk

````

