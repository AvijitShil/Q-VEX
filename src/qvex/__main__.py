"""Q-VEX CLI Entrypoint & Cosmic Knowledge Graph Banner.
Run via: python -m qvex
"""

import sys
from qvex.banner import show_banner

BOLD = "\033[1m"
CYAN = "\033[96m"
BLUE = "\033[94m"
DIM = "\033[2m"
RESET = "\033[0m"

def main() -> None:
    show_banner()
    print(f"\n{BOLD}🚀 Quickstart Guide:{RESET}")
    print(f"  {CYAN}from qvex import QVEX{RESET}")
    print(f"  {CYAN}db = QVEX(dim=384, storage_dir='./my_graph', bit_width=4){RESET}")
    print(f"  {CYAN}doc_id = db.add('Transformers use self-attention.', vector=emb){RESET}")
    print(f"  {CYAN}results = db.search('attention', vector=query_vec, k=5, hops=2){RESET}")
    print(f"\n{DIM}Documentation & Repository:{RESET} {BLUE}https://github.com/your-username/qvex{RESET}\n")

if __name__ == "__main__":
    main()
