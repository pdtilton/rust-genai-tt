use crate::adapter::adapters::support::{StreamerCapturedData, StreamerOptions};
use crate::adapter::gemini::GeminiAdapter;
use crate::adapter::inter_stream::{InterStreamEnd, InterStreamEvent};
use crate::chat::{ChatOptionsSet, ChatResponse};
use crate::webc::WebStream;
use crate::{Error, ModelIden, Result};
use std::pin::Pin;
use std::task::{Context, Poll};

use super::ContentResponse;

pub struct GeminiStreamer {
	inner: WebStream,
	options: StreamerOptions,

	// -- Set by the poll_next
	/// Flag to not poll the EventSource after a MessageStop event.
	done: bool,
	captured_data: StreamerCapturedData,
}

impl GeminiStreamer {
	pub fn new(inner: WebStream, model_iden: ModelIden, options_set: ChatOptionsSet<'_, '_>) -> Self {
		Self {
			inner,
			done: false,
			options: StreamerOptions::new(model_iden, options_set),
			captured_data: Default::default(),
		}
	}
}

// Implement futures::Stream for InterStream<GeminiStream>
impl futures::Stream for GeminiStreamer {
	type Item = Result<InterStreamEvent>;

	fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		if self.done {
			return Poll::Ready(None);
		}

		while let Poll::Ready(item) = Pin::new(&mut self.inner).poll_next(cx) {
			match item {
				Some(Ok(raw_message)) => {
					// This is the message sent by the WebStream in PrettyJsonArray mode.
					// - `[` document start
					// - `{...}` block
					// - `]` document end

					let inter_event = match raw_message.as_str() {
						"[" => InterStreamEvent::Start,
						"]" => {
							let inter_stream_end = InterStreamEnd {
								captured_usage: self.captured_data.usage.take(),
								captured_content: self.captured_data.content.take(),
								captured_reasoning_content: self.captured_data.reasoning_content.take(),
							};

							InterStreamEvent::End(inter_stream_end)
						}
						block_string => {
							// -- Parse the block to JSON
							let json_block =
								match serde_json::from_str::<ContentResponse>(block_string).map_err(|serde_error| {
									Error::StreamParse {
										model_iden: self.options.model_iden.clone(),
										serde_error,
									}
								}) {
									Ok(json_block) => json_block,
									Err(err) => {
										tracing::error!("Gemini Adapter Stream Error: {}", err);
										return Poll::Ready(Some(Err(err)));
									}
								};

							// -- Extract the Gemini Response
							let model_iden = self.options.model_iden.clone();

							let provider_model_iden =
								model_iden.with_name_or_clone(Some(json_block.model_version.clone()));

							let content =
								match GeminiAdapter::body_to_gemini_chat_response(&model_iden.clone(), &json_block) {
									Ok(content) => content,
									Err(err) => {
										tracing::error!("Gemini Adapter Stream Error: {}", err);
										return Poll::Ready(Some(Err(err)));
									}
								};

							let usage = GeminiAdapter::into_usage(&json_block.usage_metadata);

							// NOTE: Apparently in the Gemini API, all events have cumulative usage,
							//       meaning each message seems to include the tokens for all previous streams.
							//       Thus, we do not need to add it; we only need to replace captured_data.usage with the latest one.
							//       See https://twitter.com/jeremychone/status/1813734565967802859 for potential additional information.
							if self.options.capture_usage {
								self.captured_data.usage = Some(usage.clone());
							}

							InterStreamEvent::Chunk(ChatResponse {
								content,
								reasoning_content: None,
								model_iden,
								provider_model_iden,
								usage,
							})
						}
					};

					return Poll::Ready(Some(Ok(inter_event)));
				}
				Some(Err(err)) => {
					tracing::error!("Gemini Adapter Stream Error: {}", err);
					return Poll::Ready(Some(Err(Error::WebStream {
						model_iden: self.options.model_iden.clone(),
						cause: err.to_string(),
					})));
				}
				None => {
					self.done = true;
					return Poll::Ready(None);
				}
			}
		}
		Poll::Pending
	}
}
