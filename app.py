from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from functools import partial
import webbrowser


HOST = "127.0.0.1"
PORT = 8000
WEB_DIR = Path(__file__).parent / "web"


def main() -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(WEB_DIR))
    server = ThreadingHTTPServer((HOST, PORT), handler)
    url = f"http://{HOST}:{PORT}"
    print(f"猜拳必勝魔法已啟動：{url}")
    print("按 Ctrl+C 結束伺服器。")
    try:
        webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n伺服器已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
