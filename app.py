import errno
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from functools import partial
import sys
import webbrowser


HOST = "127.0.0.1"
PORT = 8000
WEB_DIR = Path(__file__).parent / "web"


class NoCacheRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    handler = partial(NoCacheRequestHandler, directory=str(WEB_DIR))
    try:
        server = ThreadingHTTPServer((HOST, PORT), handler)
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            print(
                f"無法啟動：連接埠 {PORT} 已被使用。請關閉其他伺服器，或修改 app.py 的 PORT 設定。",
                file=sys.stderr,
            )
            return
        raise

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
