use std::collections::VecDeque;
use std::env;
use std::fs;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

const DEFAULT_WORKERS: usize = 8;
const DEFAULT_QUEUE_CAPACITY: usize = 64;
const DEFAULT_REQUEST_LIMIT: usize = 16 * 1024;
const MAX_TAIL_LINES: usize = 100_000;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone)]
struct Config {
    bind: String,
    log_path: PathBuf,
    default_tail_lines: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8787".to_string(),
            log_path: PathBuf::from("training/training.log"),
            default_tail_lines: 200,
        }
    }
}

fn main() -> io::Result<()> {
    let config = parse_args(env::args().collect())?;
    let listener = TcpListener::bind(&config.bind)?;
    let shared = Arc::new(config);

    eprintln!(
        "training_log_server listening on http://{} (log path: {})",
        shared.bind,
        shared.log_path.display()
    );

    let (connections, pending) = mpsc::sync_channel::<TcpStream>(DEFAULT_QUEUE_CAPACITY);
    let pending = Arc::new(Mutex::new(pending));
    for _ in 0..DEFAULT_WORKERS {
        let config = Arc::clone(&shared);
        let pending = Arc::clone(&pending);
        thread::spawn(move || loop {
            let stream = {
                let receiver = pending.lock().expect("connection queue lock poisoned");
                receiver.recv()
            };
            let Ok(stream) = stream else { break };
            if let Err(err) = handle_connection(stream, &config) {
                eprintln!("connection error: {err}");
            }
        });
    }

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                if let Err(mpsc::TrySendError::Full(mut stream)) = connections.try_send(stream) {
                    let _ = configure_stream(&stream);
                    let _ = write_response(
                        &mut stream,
                        503,
                        "Service Unavailable",
                        "connection capacity reached\n",
                    );
                }
            }
            Err(err) => eprintln!("accept error: {err}"),
        }
    }

    Ok(())
}

fn parse_args(args: Vec<String>) -> io::Result<Config> {
    let mut config = Config::default();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--bind" => {
                i += 1;
                let value = args.get(i).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--bind needs an address")
                })?;
                config.bind = value.clone();
            }
            "--log-path" => {
                i += 1;
                let value = args.get(i).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--log-path needs a path")
                })?;
                config.log_path = PathBuf::from(value);
            }
            "--tail-lines" => {
                i += 1;
                let value = args.get(i).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--tail-lines needs a number")
                })?;
                config.default_tail_lines = value.parse::<usize>().map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "--tail-lines must be an integer",
                    )
                })?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown argument: {other}"),
                ));
            }
        }
        i += 1;
    }
    Ok(config)
}

fn print_usage() {
    eprintln!("Usage: training_log_server [--bind ADDR] [--log-path PATH] [--tail-lines N]");
    eprintln!("Defaults: --bind 127.0.0.1:8787 --log-path training/training.log --tail-lines 200");
    eprintln!("Public exposure requires an explicit --bind address and network access controls.");
    eprintln!("Routes:");
    eprintln!("  GET /health");
    eprintln!("  GET /tail");
    eprintln!("  GET /tail?lines=N");
    eprintln!("  GET /log");
}

fn configure_stream(stream: &TcpStream) -> io::Result<()> {
    stream.set_read_timeout(Some(DEFAULT_TIMEOUT))?;
    stream.set_write_timeout(Some(DEFAULT_TIMEOUT))
}

fn handle_connection(mut stream: TcpStream, config: &Config) -> io::Result<()> {
    configure_stream(&stream)?;
    let (method, path) = match read_request(&mut stream, DEFAULT_REQUEST_LIMIT) {
        Ok(Some(request)) => request,
        Ok(None) => return Ok(()),
        Err(err) if matches!(err.kind(), io::ErrorKind::InvalidData) => {
            return write_response(&mut stream, 400, "Bad Request", &format!("{err}\n"));
        }
        Err(err) => return Err(err),
    };

    if method != "GET" {
        return write_response(
            &mut stream,
            405,
            "Method Not Allowed",
            "only GET is supported\n",
        );
    }

    if path == "/health" {
        return write_response(&mut stream, 200, "OK", "ok\n");
    }

    if path.starts_with("/tail") {
        let requested_lines =
            parse_query_usize(&path, "lines").unwrap_or(config.default_tail_lines);
        let requested_lines = match validate_tail_lines(requested_lines) {
            Ok(lines) => lines,
            Err(message) => {
                return write_response(&mut stream, 400, "Bad Request", &format!("{message}\n"));
            }
        };
        return match read_tail_lines(&config.log_path, requested_lines) {
            Ok(content) => write_response(&mut stream, 200, "OK", &content),
            Err(err) => write_response(
                &mut stream,
                500,
                "Internal Server Error",
                &format!("failed to read tail: {err}\n"),
            ),
        };
    }

    if path == "/log" {
        return match fs::read_to_string(&config.log_path) {
            Ok(content) => write_response(&mut stream, 200, "OK", &content),
            Err(err) => write_response(
                &mut stream,
                500,
                "Internal Server Error",
                &format!("failed to read log: {err}\n"),
            ),
        };
    }

    write_response(
        &mut stream,
        404,
        "Not Found",
        "routes: /health, /tail, /tail?lines=N, /log\n",
    )
}

fn read_request<R: Read>(reader: R, limit: usize) -> io::Result<Option<(String, String)>> {
    let mut reader = BufReader::new(reader.take((limit + 1) as u64));
    let mut request_line = String::new();
    let mut total = reader.read_line(&mut request_line)?;
    if total == 0 {
        return Ok(None);
    }
    if total > limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "request is too large",
        ));
    }

    let mut terminated = false;
    loop {
        let mut header_line = String::new();
        let read = reader.read_line(&mut header_line)?;
        total += read;
        if total > limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "request is too large",
            ));
        }
        if read == 0 {
            break;
        }
        if header_line == "\r\n" || header_line == "\n" {
            terminated = true;
            break;
        }
    }
    if !terminated {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "request headers are incomplete",
        ));
    }

    parse_request_line(&request_line)
        .map(Some)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "could not parse request line"))
}

fn parse_request_line(line: &str) -> Option<(String, String)> {
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let _version = parts.next()?;
    Some((method, path))
}

fn parse_query_usize(path: &str, key: &str) -> Option<usize> {
    let (_, query) = path.split_once('?')?;
    for pair in query.split('&') {
        let (k, value) = pair.split_once('=')?;
        if k == key {
            return value.parse::<usize>().ok();
        }
    }
    None
}

fn validate_tail_lines(lines: usize) -> Result<usize, String> {
    if lines <= MAX_TAIL_LINES {
        Ok(lines)
    } else {
        Err(format!("lines must be at most {MAX_TAIL_LINES}"))
    }
}

fn read_tail_lines(path: &PathBuf, lines: usize) -> io::Result<String> {
    if lines == 0 {
        return Ok(String::new());
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut tail = VecDeque::with_capacity(lines.min(10_000));

    for line in reader.lines() {
        let line = line?;
        if tail.len() == lines {
            tail.pop_front();
        }
        tail.push_back(line);
    }

    let mut out = String::new();
    for line in tail {
        out.push_str(&line);
        out.push('\n');
    }

    Ok(out)
}

fn write_response(stream: &mut TcpStream, code: u16, reason: &str, body: &str) -> io::Result<()> {
    let headers = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        code,
        reason,
        body.len()
    );
    stream.write_all(headers.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn default_bind_is_loopback_only() {
        assert_eq!(Config::default().bind, "127.0.0.1:8787");
    }

    #[test]
    fn reads_complete_request_within_limit() {
        let request = b"GET /tail?lines=10 HTTP/1.1\r\nHost: localhost\r\n\r\n";
        let parsed = read_request(Cursor::new(request), request.len()).unwrap();
        assert_eq!(
            parsed,
            Some(("GET".to_owned(), "/tail?lines=10".to_owned()))
        );
    }

    #[test]
    fn rejects_oversized_request() {
        let request = format!("GET / HTTP/1.1\r\nX-Test: {}\r\n\r\n", "x".repeat(128));
        let err = read_request(Cursor::new(request), 64).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn rejects_incomplete_headers() {
        let err =
            read_request(Cursor::new(b"GET / HTTP/1.1\r\nHost: localhost\r\n"), 128).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn tail_line_limit_bounds_memory_use() {
        assert_eq!(validate_tail_lines(MAX_TAIL_LINES), Ok(MAX_TAIL_LINES));
        assert!(validate_tail_lines(MAX_TAIL_LINES + 1).is_err());
    }
}
