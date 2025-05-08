//! Printer utility to help print a chat stream
//! > Note: This is primarily for quick testing and temporary debugging

use crate::chat::{ChatStreamEvent, ChatStreamResponse};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncWriteExt as _, Stdout};

// Note: This module has its own Error type (see end of file)
type Result<T> = core::result::Result<T, Error>;

// region:    --- PrintChatOptions

/// Options to be passed into the `printer::print_chat_stream`
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PrintChatStreamOptions {
	print_events: Option<bool>,
}

/// Constructors
impl PrintChatStreamOptions {
	/// Create a `PrintChatStreamOptions` with the `print_events` field set to `true`
	pub fn from_print_events(print_events: bool) -> Self {
		PrintChatStreamOptions {
			print_events: Some(print_events),
		}
	}
}

// endregion: --- PrintChatOptions

/// Convenient function that prints a chat stream, captures the content, and returns it (only concatenated chunks).
pub async fn print_chat_stream(
	chat_res: ChatStreamResponse,
	options: Option<&PrintChatStreamOptions>,
) -> Result<String> {
	let mut stdout = tokio::io::stdout();
	let res = print_chat_stream_inner(&mut stdout, chat_res, options).await;
	// Ensure tokio stdout flush is called, regardless of success or failure.
	stdout.flush().await?;
	res
}

async fn print_chat_stream_inner(
	stdout: &mut Stdout,
	chat_res: ChatStreamResponse,
	options: Option<&PrintChatStreamOptions>,
) -> Result<String> {
	let mut stream = chat_res.stream;

	let mut content_capture = String::new();

	let print_events = options.and_then(|o| o.print_events).unwrap_or_default();

	while let Some(Ok(stream_event)) = stream.next().await {
		let event_messages = {
			let mut messages = Vec::new();

			match stream_event {
				ChatStreamEvent::Start => {
					if print_events {
						// TODO: Might implement pretty JSON formatting
						messages.push(("\n-- ChatStreamEvent::Start\n".to_string(), None))
					}
				}
				ChatStreamEvent::Chunk(chunk) => {
					let mut content = Vec::new();

					for msg_content in chunk.content {
						match msg_content {
							super::MessageContent::Text(text) => {
								content.push(text);
							}
							super::MessageContent::Parts(content_parts) => {
								for part in content_parts {
									match part {
										super::ContentPart::Text(text) => {
											content.push(text);
										}
										super::ContentPart::Image { content_type, source } => {
											let event_info =
												format!("\n-- ChatStreamEvent InlineData {}\n", content_type);
											let msg = match source {
												super::ImageSource::Base64(data) => {
													format!("Base64: {data}\n")
												}
												super::ImageSource::Url(url) => {
													format!("url: {url}\n")
												}
											};
											messages.push((event_info, Some(msg)))
										}
									}
								}
							}
							super::MessageContent::ToolCalls(tool_calls) => {
								for tool_call in tool_calls {
									messages.push((
										"\n-- ChatStreamEvent Tool Call\n".to_string(),
										Some(format!(
											"id: {} fn: {} args:{}",
											tool_call.call_id, tool_call.fn_name, tool_call.fn_arguments
										)),
									));
								}
							}
							super::MessageContent::ToolResponses(tool_responses) => {
								for tool_response in tool_responses {
									messages.push((
										"\n-- ChatStreamEvent Tool Call Response\n".to_string(),
										Some(format!(
											"id: {} result: {}",
											tool_response.call_id, tool_response.content
										)),
									));
								}
							}
						}
					}

					if !content.is_empty() {
						messages.push((format!("\n-- ChatStreamEvent Text\n"), Some(content.join(" "))));
					}

					if let Some(reasoning_content) = chunk.reasoning_content {
						messages.push((
							format!("\n-- ChatStreamEvent Reasoning Text\n"),
							Some(reasoning_content),
						));
					}
				}

				ChatStreamEvent::End(end_event) => {
					if print_events {
						// TODO: Might implement pretty JSON formatting
						messages.push((format!("\n\n-- ChatStreamEvent::End {end_event:?}\n"), None));
					}
				}
			}

			messages
		};

		for (event_info, content) in event_messages {
			stdout.write_all(event_info.as_bytes()).await?;
			if let Some(content) = content {
				content_capture.push_str(&content);
				stdout.write_all(content.as_bytes()).await?;
			}
		}

		stdout.flush().await?;
	}

	stdout.write_all(b"\n").await?;

	Ok(content_capture)
}

// region:    --- Error

// Note 1: The printer has its own error type because it is more of a utility, and therefore
//         making the main crate error aware of the different error types would be unnecessary.
//
// Note 2: This Printer Error is not wrapped in the main crate error because the printer
//         functions are not used by any other crate functions (they are more of a debug utility)

use derive_more::From;

/// The Printer error.
#[derive(Debug, From)]
pub enum Error {
	/// The `tokio::io::Error` when using `tokio::io::stdout`
	#[from]
	TokioIo(tokio::io::Error),
}

// region:    --- Error Boilerplate

impl core::fmt::Display for Error {
	fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::result::Result<(), core::fmt::Error> {
		write!(fmt, "{self:?}")
	}
}

impl std::error::Error for Error {}

// endregion: --- Error Boilerplate
