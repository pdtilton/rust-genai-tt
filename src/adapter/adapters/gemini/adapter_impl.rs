use crate::adapter::adapters::support::get_api_key;
use crate::adapter::gemini::GeminiStreamer;
use crate::adapter::{Adapter, AdapterKind, ServiceType, WebRequestData};
use crate::chat::{
	ChatOptionsSet, ChatRequest, ChatResponse, ChatResponseFormat, ChatRole, ChatStream, ChatStreamResponse,
	ContentPart, ImageSource, MessageContent, ToolCall, ToolResponse, Usage,
};
use crate::resolver::{AuthData, Endpoint};
use crate::webc::{WebResponse, WebStream};
use crate::{Error, ModelIden, Result, ServiceTarget};
use reqwest::RequestBuilder;
use serde::Deserialize;
use serde_json::{Value, json};
use value_ext::JsonValueExt;

pub struct GeminiAdapter;

const MODELS: &[&str] = &["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro"];

// curl \
//   -H 'Content-Type: application/json' \
//   -d '{"contents":[{"parts":[{"text":"Explain how AI works"}]}]}' \
//   -X POST 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=YOUR_API_KEY'

impl GeminiAdapter {
	pub const API_KEY_DEFAULT_ENV_NAME: &str = "GEMINI_API_KEY";
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Blob {
	mime_type: String,
	data: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FileData {
	mime_type: String,
	file_uri: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FunctionCall {
	id: String,
	name: String,
	args: Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FunctionResponse {
	id: String,
	name: String,
	response: Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum Language {
	Python,
	LanguageUnspecified,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExecutableCode {
	language: Language,
	code: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
enum Outcome {
	OutcomeUnspecified,
	OutcomeOk,
	OutcomeFailed,
	OutcomeDeadlineExceeded,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodeExecutionResult {
	outcome: Outcome,
	output: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
enum Part {
	Tought(()),
	// data
	Text(String),
	InlineData(Blob),
	FunctionCall(FunctionCall),
	FunctionRresponse(FunctionResponse),
	FileData(FileData),
	ExecutableCode(ExecutableCode),
	CodeExecutionResult(CodeExecutionResult),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Content {
	#[serde(default)]
	parts: Vec<Part>,
	role: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum FinishReason {
	FinishReasonUnspecified,
	Stop,
	MaxTokens,
	Safety,
	Recitation,
	Language,
	Other,
	BlockList,
	ProhibitedContent,
	Spii,
	MalformedFunctionCall,
	ImageSafety,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum HarmCategory {
	HarmCategoryHarassment,
	HarmCategoryHateSpeech,
	HarmCategorySexuallyExplicit,
	HarmCategoryDangerous,
	HarmCategoryCivicIntegrity,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum HarmProbability {
	HarmProbabilityUnspecified,
	Negligible,
	Low,
	Medium,
	High,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmBlockThreshold {
	BlockNone,
	BlockOnlyHigh,
	BlockMediumAndAbove,
	BlockLowAndAbove,
	HarmBlockThresholdUnspecified,
	Off,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SafetyRating {
	category: HarmCategory,
	probability: HarmProbability,
	blocked: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CitationSource {
	#[serde(default)]
	start_index: Option<u32>,
	#[serde(default)]
	end_index: Option<u32>,
	#[serde(default)]
	uri: Option<String>,
	#[serde(default)]
	license: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CitationMetadata {
	#[serde(default)]
	citation_sources: Vec<CitationSource>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GroundingPassageId {
	passage_id: String,
	part_index: i32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SemanticRetrieverChunk {
	source: String,
	chunk: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AttributionSourceId {
	// source
	grounding_passage: GroundingPassageId,
	semantic_retriever_chunk: SemanticRetrieverChunk,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GroundingAttribution {
	source_id: AttributionSourceId,
	content: Content,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Web {
	uri: String,
	title: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GroundingChunk {
	web: Web,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Segment {
	part_index: i32,
	start_index: i32,
	end_index: i32,
	text: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GroundingSupport {
	#[serde(default)]
	grounding_chunk_indices: Vec<i32>,
	#[serde(default)]
	confidence_scores: Vec<f32>,
	segment: Segment,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchEntryPoint {
	#[serde(default)]
	rendered_content: Option<String>,
	#[serde(default)]
	sdk_blob: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RetrievalMetadata {
	#[serde(default)]
	google_search_dynamic_retrieval_score: Option<f32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GroundingMetadata {
	#[serde(default)]
	grounding_chunks: Vec<GroundingChunk>,
	#[serde(default)]
	grounding_supports: Vec<GroundingSupport>,
	#[serde(default)]
	web_search_queries: Vec<String>,
	#[serde(default)]
	search_entry_point: Option<SearchEntryPoint>,
	retrieval_metadata: RetrievalMetadata,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename = "candidate")]
struct LogCandidate {
	token: String,
	token_id: i32,
	log_probability: f32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TopCandidates {
	candidates: LogCandidate,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LogprobsResult {
	#[serde(default)]
	top_candidates: Vec<TopCandidates>,
	#[serde(default)]
	chosen_candidates: Vec<LogCandidate>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UrlRetrievalContext {
	retrieved_url: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UrlRetrievalMetadata {
	#[serde(default)]
	url_retrieval_contexts: Vec<UrlRetrievalContext>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Candidate {
	content: Content,
	#[serde(default)]
	finish_reason: Option<FinishReason>,
	#[serde(default)]
	safety_ratings: Vec<SafetyRating>,
	#[serde(default)]
	citation_metadata: Option<CitationMetadata>,
	#[serde(default)]
	grounding_attributions: Vec<GroundingAttribution>,
	#[serde(default)]
	grounding_metadata: Option<GroundingMetadata>,
	#[serde(default)]
	avg_logprobs: Option<f32>,
	#[serde(default)]
	logprobs_result: Option<LogprobsResult>,
	#[serde(default)]
	url_retrieval_metadata: Option<UrlRetrievalMetadata>,
	#[serde(default)]
	index: Option<i32>,
	#[serde(default)]
	token_count: Option<i32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum BlockReason {
	BlockReasonUnspecified,
	Safety,
	Other,
	BlockList,
	ProhibitedContent,
	ImageSafety,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct PromptFeedBack {
	#[serde(default)]
	block_reason: Option<BlockReason>,
	#[serde(default)]
	safety_ratings: Vec<SafetyRating>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum Modality {
	ModalityUnspecified,
	Text,
	Image,
	Audio,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ModalityTokenCount {
	modality: Modality,
	token_count: i32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct UsageMetadata {
	prompt_token_count: i32,
	#[serde(default)]
	cached_content_token_count: Option<i32>,
	candidates_token_count: i32,
	#[serde(default)]
	tool_use_prompt_token_count: Option<i32>,
	#[serde(default)]
	thoughts_token_count: Option<i32>,
	total_token_count: i32,
	#[serde(default)]
	prompt_tokens_details: Vec<ModalityTokenCount>,
	#[serde(default)]
	cache_tokens_details: Vec<ModalityTokenCount>,
	#[serde(default)]
	candidates_tokens_details: Vec<ModalityTokenCount>,
	#[serde(default)]
	tool_use_prompt_tokens_details: Vec<ModalityTokenCount>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ContentResponse {
	#[serde(default)]
	pub(super) candidates: Vec<Candidate>,
	#[serde(default)]
	pub(super) prompt_feedback: Option<PromptFeedBack>,
	pub(super) usage_metadata: UsageMetadata,
	pub(super) model_version: String,
	#[serde(default)]
	pub(super) error: Option<Value>,
}

impl Adapter for GeminiAdapter {
	fn default_endpoint() -> Endpoint {
		const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/";
		Endpoint::from_static(BASE_URL)
	}

	fn default_auth() -> AuthData {
		AuthData::from_env(Self::API_KEY_DEFAULT_ENV_NAME)
	}

	/// Note: For now, this returns the common models (see above)
	async fn all_model_names(_kind: AdapterKind) -> Result<Vec<String>> {
		Ok(MODELS.iter().map(|s| s.to_string()).collect())
	}

	/// NOTE: As Google Gemini has decided to put their API_KEY in the URL,
	///       this will return the URL without the API_KEY in it. The API_KEY will need to be added by the caller.
	fn get_service_url(model: &ModelIden, service_type: ServiceType, endpoint: Endpoint) -> String {
		let base_url = endpoint.base_url();
		let model_name = model.model_name.clone();
		match service_type {
			ServiceType::Chat => format!("{base_url}models/{model_name}:generateContent"),
			ServiceType::ChatStream => format!("{base_url}models/{model_name}:streamGenerateContent"),
		}
	}

	fn to_web_request_data(
		target: ServiceTarget,
		service_type: ServiceType,
		chat_req: ChatRequest,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		let ServiceTarget { endpoint, auth, model } = target;

		// -- api_key
		let api_key = get_api_key(auth, &model)?;

		// -- url
		// NOTE: Somehow, Google decided to put the API key in the URL.
		//       This should be considered an antipattern from a security point of view
		//       even if it is done by the well respected Google. Everybody can make mistake once in a while.
		// e.g., '...models/gemini-1.5-flash-latest:generateContent?key=YOUR_API_KEY'
		let url = Self::get_service_url(&model, service_type, endpoint);
		let url = format!("{url}?key={api_key}");

		// -- parts
		let GeminiChatRequestParts {
			system,
			contents,
			tools,
			safety_settings,
		} = Self::into_gemini_request_parts(model, chat_req)?;

		// -- Playload
		let mut payload = json!({
			"contents": contents,
		});

		// -- headers (empty for gemini, since API_KEY is in url)
		let headers = vec![];

		// Note: It's unclear from the spec if the content of systemInstruction should have a role.
		//       Right now, it is omitted (since the spec states it can only be "user" or "model")
		//       It seems to work. https://ai.google.dev/api/rest/v1beta/models/generateContent
		if let Some(system) = system {
			payload.x_insert(
				"systemInstruction",
				json!({
					"parts": [ { "text": system }]
				}),
			)?;
		}

		// -- Tools
		if let Some(tools) = tools {
			payload.x_insert(
				"tools",
				json!({
					"function_declarations": tools
				}),
			)?;
		}

		if let Some(safety_settings) = safety_settings {
			payload.x_insert("safetySettings", json!(safety_settings))?;
		}

		// -- Response Format
		if let Some(ChatResponseFormat::JsonSpec(st_json)) = options_set.response_format() {
			// x_insert
			//     responseMimeType: "application/json",
			// responseSchema: {
			payload.x_insert("/generationConfig/responseMimeType", "application/json")?;
			let mut schema = st_json.schema.clone();
			schema.x_walk(|parent_map, name| {
				if name == "additionalProperties" {
					parent_map.remove("additionalProperties");
				}
				true
			});
			payload.x_insert("/generationConfig/responseSchema", schema)?;
		}

		// -- Add supported ChatOptions
		if let Some(temperature) = options_set.temperature() {
			payload.x_insert("/generationConfig/temperature", temperature)?;
		}

		if !options_set.stop_sequences().is_empty() {
			payload.x_insert("/generationConfig/stopSequences", options_set.stop_sequences())?;
		}

		if let Some(max_tokens) = options_set.max_tokens() {
			payload.x_insert("/generationConfig/maxOutputTokens", max_tokens)?;
		}
		if let Some(top_p) = options_set.top_p() {
			payload.x_insert("/generationConfig/topP", top_p)?;
		}
		if !options_set.response_modality().is_empty() {
			payload.x_insert(
				"/generationConfig/responseModalities",
				json!(options_set.response_modality()),
			)?;
		}

		println!("headers: {:?}", headers);
		println!("sending payload: {:?}", payload);

		Ok(WebRequestData { url, headers, payload })
	}

	fn to_chat_response(
		model_iden: ModelIden,
		web_response: WebResponse,
		_options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatResponse> {
		let WebResponse { body, .. } = web_response;

		println!("body: {:?}", body);

		let body: ContentResponse = serde_json::from_value(body)?;

		// -- Capture the provider_model_iden
		// TODO: Need to be implemented (if available), for now, just clone model_iden
		let provider_model_iden = model_iden.with_name_or_clone(Some(body.model_version.clone()));

		let content = Self::body_to_gemini_chat_response(&model_iden.clone(), &body)?;
		let usage = Self::into_usage(&body.usage_metadata);

		println!("content: {:?}", content);
		println!("Usage: {:?}", usage);

		Ok(ChatResponse {
			content,
			reasoning_content: None,
			model_iden,
			provider_model_iden,
			usage,
		})
	}

	fn to_chat_stream(
		model_iden: ModelIden,
		reqwest_builder: RequestBuilder,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatStreamResponse> {
		let web_stream = WebStream::new_with_pretty_json_array(reqwest_builder);

		let gemini_stream = GeminiStreamer::new(web_stream, model_iden.clone(), options_set);
		let chat_stream = ChatStream::from_inter_stream(gemini_stream);

		Ok(ChatStreamResponse {
			model_iden,
			stream: chat_stream,
		})
	}
}

// region:    --- Support

/// Support functions for GeminiAdapter
impl GeminiAdapter {
	pub(super) fn body_to_gemini_chat_response(
		model_iden: &ModelIden,
		body: &ContentResponse,
	) -> Result<Vec<MessageContent>> {
		// If the body has an `error` property, then it is assumed to be an error.
		if let Some(error) = &body.error {
			return Err(Error::StreamEventError {
				model_iden: model_iden.clone(),
				body: error.clone(),
			});
		}

		//let mut response = body.x_take::<Value>("/candidates/0/content/parts/0")?;
		let mut content = vec![];

		for entry in &body.candidates {
			let mut tool_responses = vec![];
			let mut tool_calls = vec![];
			let mut parts = vec![];

			for candidate in &entry.content.parts {
				match candidate {
					Part::Text(text) => parts.push(ContentPart::Text(text.clone())),
					Part::Tought(_) => {
						tracing::warn!("Thought not implemented");
					}
					Part::InlineData(blob) => {
						if blob.mime_type.starts_with("image") {
							parts.push(ContentPart::Image {
								content_type: blob.mime_type.clone(),
								source: ImageSource::Base64(blob.data.clone().into()),
							})
						}
					}
					Part::FunctionCall(function_call) => {
						tool_calls.push(ToolCall {
							call_id: function_call.id.clone(),
							fn_name: function_call.name.clone(),
							fn_arguments: function_call.args.clone(),
						});
					}
					Part::FunctionRresponse(function_response) => {
						tool_responses.push(ToolResponse {
							call_id: function_response.id.clone(),
							content: function_response.response.to_string(),
						});
					}
					Part::FileData(file_data) => {
						if file_data.mime_type.starts_with("image") {
							parts.push(ContentPart::Image {
								content_type: file_data.mime_type.clone(),
								source: ImageSource::Url(file_data.file_uri.clone()),
							})
						}
					}
					Part::ExecutableCode(_executable_code) => {
						tracing::warn!("Executable code not implemented");
					}
					Part::CodeExecutionResult(_code_execution_result) => {
						tracing::warn!("Code execution result not implemented");
					}
				}
			}

			if !parts.is_empty() {
				content.push(MessageContent::Parts(parts));
			}

			if !tool_calls.is_empty() {
				content.push(MessageContent::ToolCalls(tool_calls));
			}

			if !tool_responses.is_empty() {
				content.push(MessageContent::ToolResponses(tool_responses));
			}
		}

		Ok(content)
	}

	pub(super) fn into_usage(usage_value: &UsageMetadata) -> Usage {
		let prompt_tokens: Option<i32> = Some(usage_value.prompt_token_count);
		let completion_tokens: Option<i32> = Some(usage_value.candidates_token_count);
		let total_tokens: Option<i32> = Some(usage_value.total_token_count);

		Usage {
			prompt_tokens,
			// for now, None for Gemini
			prompt_tokens_details: None,

			completion_tokens,
			// for now, None for Gemini
			completion_tokens_details: None,

			total_tokens,
		}
	}

	/// Takes the genai ChatMessages and builds the System string and JSON Messages for Gemini.
	/// - Role mapping `ChatRole:User -> role: "user"`, `ChatRole::Assistant -> role: "model"`
	/// - `ChatRole::System` is concatenated (with an empty line) into a single `system` for the system instruction.
	///   - This adapter uses version v1beta, which supports `systemInstruction`
	/// - The eventual `chat_req.system` is pushed first into the "systemInstruction"
	fn into_gemini_request_parts(model_iden: ModelIden, chat_req: ChatRequest) -> Result<GeminiChatRequestParts> {
		let mut contents: Vec<Value> = Vec::new();
		let mut systems: Vec<String> = Vec::new();

		if let Some(system) = chat_req.system {
			systems.push(system);
		}

		// -- Build
		for msg in chat_req.messages {
			match msg.role {
				// For now, system goes as "user" (later, we might have adapter_config.system_to_user_impl)
				ChatRole::System => {
					let MessageContent::Text(content) = msg.content else {
						return Err(Error::MessageContentTypeNotSupported {
							model_iden,
							cause: "Only MessageContent::Text supported for this model (for now)",
						});
					};
					systems.push(content)
				}
				ChatRole::User => {
					let content = match msg.content {
						MessageContent::Text(content) => json!([{"text": content}]),
						MessageContent::Parts(parts) => {
							json!(
								parts
									.iter()
									.map(|part| match part {
										ContentPart::Text(text) => json!({"text": text.clone()}),
										ContentPart::Image { content_type, source } => {
											match source {
												ImageSource::Url(url) => json!({
													"file_data": {
														"mime_type": content_type,
														"file_uri": url
													}
												}),
												ImageSource::Base64(content) => json!({
													"inline_data": {
														"mime_type": content_type,
														"data": content
													}
												}),
											}
										}
									})
									.collect::<Vec<Value>>()
							)
						}
						MessageContent::ToolCalls(tool_calls) => {
							json!(
								tool_calls
									.into_iter()
									.map(|tool_call| {
										json!({
											"functionCall": {
												"name": tool_call.fn_name,
												"args": tool_call.fn_arguments,
											}
										})
									})
									.collect::<Vec<Value>>()
							)
						}
						MessageContent::ToolResponses(tool_responses) => {
							json!(
								tool_responses
									.into_iter()
									.map(|tool_response| {
										json!({
											"functionResponse": {
												"name": tool_response.call_id,
												"response": {
													"name": tool_response.call_id,
													"content": serde_json::from_str(&tool_response.content).unwrap_or(Value::Null),
												}
											}
										})
									})
									.collect::<Vec<Value>>()
							)
						}
					};

					contents.push(json!({"role": "user", "parts": content}));
				}
				ChatRole::Assistant => {
					match msg.content {
						MessageContent::Text(content) => {
							contents.push(json!({"role": "model", "parts": [{"text": content}]}))
						}
						MessageContent::ToolCalls(tool_calls) => contents.push(json!({
							"role": "model",
							"parts": tool_calls
								.into_iter()
								.map(|tool_call| {
									json!({
										"functionCall": {
											"name": tool_call.fn_name,
											"args": tool_call.fn_arguments,
										}
									})
								})
								.collect::<Vec<Value>>()
						})),
						_ => {
							return Err(Error::MessageContentTypeNotSupported {
								model_iden,
								cause: "Only MessageContent::Text and MessageContent::ToolCalls supported for this model (for now)",
							});
						}
					};
				}
				ChatRole::Tool => {
					let content = match msg.content {
						MessageContent::ToolCalls(tool_calls) => {
							json!(
								tool_calls
									.into_iter()
									.map(|tool_call| {
										json!({
											"functionCall": {
												"name": tool_call.fn_name,
												"args": tool_call.fn_arguments,
											}
										})
									})
									.collect::<Vec<Value>>()
							)
						}
						MessageContent::ToolResponses(tool_responses) => {
							json!(
								tool_responses
									.into_iter()
									.map(|tool_response| {
										json!({
											"functionResponse": {
												"name": tool_response.call_id,
												"response": {
													"name": tool_response.call_id,
													"content": serde_json::from_str(&tool_response.content).unwrap_or(Value::Null),
												}
											}
										})
									})
									.collect::<Vec<Value>>()
							)
						}
						_ => {
							return Err(Error::MessageContentTypeNotSupported {
								model_iden,
								cause: "ChatRole::Tool can only be MessageContent::ToolCall or MessageContent::ToolResponse",
							});
						}
					};

					contents.push(json!({"role": "user", "parts": content}));
				}
			}
		}

		let system = if !systems.is_empty() {
			Some(systems.join("\n"))
		} else {
			None
		};

		let tools = chat_req.tools.map(|tools| {
			tools
				.into_iter()
				.map(|tool| {
					// TODO: Need to handle the error correctly
					// TODO: Needs to have a custom serializer (tool should not have to match to a provider)
					// NOTE: Right now, low probability, so, we just return null if cannot convert to value.
					json!({
						"name": tool.name,
						"description": tool.description,
						"parameters": tool.schema,
					})
				})
				.collect::<Vec<Value>>()
		});

		let safety_settings = chat_req
			.safety_settings
			.map(|safety| safety.into_iter().map(|setting| json!(setting)).collect::<Vec<Value>>());

		Ok(GeminiChatRequestParts {
			system,
			contents,
			tools,
			safety_settings,
		})
	}
}

// struct Gemini

struct GeminiChatRequestParts {
	system: Option<String>,
	/// The chat history (user and assistant, except for the last user message which is a message)
	contents: Vec<Value>,

	/// The tools to use
	tools: Option<Vec<Value>>,

	safety_settings: Option<Vec<Value>>,
}

// endregion: --- Support
