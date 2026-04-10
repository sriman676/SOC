import os

from inference import app


def main() -> None:
    import errno
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    preferred_port = int(os.getenv("PORT", "7860"))

    max_port = preferred_port + 19
    launched = False
    for port in range(preferred_port, max_port + 1):
        try:
            config = uvicorn.Config(app, host=host, port=port)
            server = uvicorn.Server(config)
            server.run()

            # If startup failed (including bind errors), server.started stays False.
            if not server.started:
                if port < max_port:
                    print(f"[WARN] Port {port} failed to start, retrying on {port + 1}.")
                continue
            launched = True
            break
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            if port < max_port:
                print(f"[WARN] Port {port} busy, retrying on {port + 1}.")
            continue
        except SystemExit as exc:
            # Uvicorn can surface bind failures as SystemExit(1).
            if exc.code in (0, None):
                launched = True
                break
            if port < max_port:
                print(f"[WARN] Port {port} failed with exit {exc.code}, retrying on {port + 1}.")
            continue

    if not launched:
        raise OSError(errno.EADDRINUSE, "No available port in fallback range")


if __name__ == "__main__":
    main()
