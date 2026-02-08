use std::collections::VecDeque;
use std::env;
use std::fs;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

#[derive(Clone)]
struct Config {
    bind: String,
    log_path: PathBuf,
    default_tail_lines: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:8787".to_string(),
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

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let config = Arc::clone(&shared);
                thread::spawn(move || {
                    if let Err(err) = handle_connection(stream, &config) {
                        eprintln!("connection error: {err}");
                    }
                });
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
    eprintln!("Defaults: --bind 0.0.0.0:8787 --log-path training/training.log --tail-lines 200");
    eprintln!("Routes:");
    eprintln!("  GET /health");
    eprintln!("  GET /tail");
    eprintln!("  GET /tail?lines=N");
    eprintln!("  GET /log");
}

fn handle_connection(mut stream: TcpStream, config: &Config) -> io::Result<()> {
    let (method, path) = {
        let mut reader = BufReader::new(&mut stream);
        let mut request_line = String::new();
        if reader.read_line(&mut request_line)? == 0 {
            return Ok(());
        }

        loop {
            let mut header_line = String::new();
            if reader.read_line(&mut header_line)? == 0 {
                break;
            }
            if header_line == "\r\n" || header_line == "\n" {
                break;
            }
        }

        match parse_request_line(&request_line) {
            Some(v) => v,
            None => {
                return write_response(
                    &mut stream,
                    400,
                    "Bad Request",
                    "could not parse request line\n",
                );
            }
        }
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
