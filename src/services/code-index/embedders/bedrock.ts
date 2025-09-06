import {
	BedrockRuntimeClient,
	InvokeModelCommand,
	InvokeModelCommandInput,
	InvokeModelCommandOutput,
} from "@aws-sdk/client-bedrock-runtime"
import { IEmbedder, EmbeddingResponse, EmbedderInfo } from "../interfaces"
import { MAX_BATCH_RETRIES, INITIAL_RETRY_DELAY_MS } from "../constants"
import { t } from "../../../i18n"
import { withValidationErrorHandling, formatEmbeddingError, HttpError } from "../shared/validation-helpers"
import { TelemetryEventName } from "@roo-code/types"
import { TelemetryService } from "@roo-code/telemetry"
import { createBedrockRuntimeClient } from "../../../api/providers/bedrock-shared"

/**
 * Configuration options for Amazon Bedrock Embedding Provider
 */
interface BedrockEmbedderConfig {
	modelId: string
	region?: string
	endpointUrl?: string
	dimensions?: number
	normalize?: boolean
	inputType?: string
	batchSize?: number
	maxConcurrency?: number
	timeoutMs?: number
	maxRetries?: number
	accessKeyId?: string
	secretAccessKey?: string
	sessionToken?: string
}

/**
 * Amazon Bedrock implementation of the embedder interface with batching and rate limiting
 */
export class AmazonBedrockEmbeddingProvider implements IEmbedder {
	private client: BedrockRuntimeClient
	private readonly config: BedrockEmbedderConfig
	private readonly maxRetries: number

	/**
	 * Creates a new Amazon Bedrock embedder
	 * @param config Configuration options for the Bedrock embedder
	 */
	constructor(config: BedrockEmbedderConfig) {
		this.config = {
			batchSize: 32,
			maxConcurrency: 5,
			maxRetries: MAX_BATCH_RETRIES,
			...config,
		}
		this.maxRetries = this.config.maxRetries!

		this.client = createBedrockRuntimeClient({
			awsRegion: this.config.region,
			awsBedrockEndpoint: this.config.endpointUrl,
			awsBedrockEndpointEnabled: !!this.config.endpointUrl,
			awsAccessKey: this.config.accessKeyId,
			awsSecretKey: this.config.secretAccessKey,
			awsSessionToken: this.config.sessionToken,
		})
	}

	/**
	 * Creates embeddings for the given texts with batching and rate limiting
	 * @param texts Array of text strings to embed
	 * @param model Optional model identifier
	 * @returns Promise resolving to embedding response
	 */
	async createEmbeddings(texts: string[], model?: string): Promise<EmbeddingResponse> {
		const modelToUse = model || this.config.modelId
		const allEmbeddings: number[][] = []
		const usage = { promptTokens: 0, totalTokens: 0 }

		if (this.isTitanModel(modelToUse)) {
			// Titan models: process one at a time with concurrency
			const results = await this.processTitanEmbeddings(texts, modelToUse)
			allEmbeddings.push(...results.embeddings)
			usage.promptTokens = results.usage.promptTokens
			usage.totalTokens = results.usage.totalTokens
		} else if (this.isCohereModel(modelToUse)) {
			// Cohere models: batch processing
			const results = await this.processCohereEmbeddings(texts, modelToUse)
			allEmbeddings.push(...results.embeddings)
			usage.promptTokens = results.usage.promptTokens
			usage.totalTokens = results.usage.totalTokens
		} else {
			throw new Error(t("embeddings:unsupportedModel", { model: modelToUse }))
		}

		return { embeddings: allEmbeddings, usage }
	}

	/**
	 * Validates the Bedrock embedder configuration by attempting a minimal embedding request
	 * @returns Promise resolving to validation result with success status and optional error message
	 */
	async validateConfiguration(): Promise<{ valid: boolean; error?: string }> {
		return withValidationErrorHandling(async () => {
			try {
				// Test with a minimal embedding request
				await this.createEmbeddings(["test"], this.config.modelId)
				return { valid: true }
			} catch (error) {
				// Capture telemetry for validation errors
				TelemetryService.instance.captureEvent(TelemetryEventName.CODE_INDEX_ERROR, {
					error: error instanceof Error ? error.message : String(error),
					stack: error instanceof Error ? error.stack : undefined,
					location: "AmazonBedrockEmbeddingProvider:validateConfiguration",
				})
				throw error
			}
		}, "bedrock")
	}

	get embedderInfo(): EmbedderInfo {
		return {
			name: "bedrock",
		}
	}

	/**
	 * Checks if the model is a Titan model
	 */
	private isTitanModel(modelId: string): boolean {
		return modelId.startsWith("amazon.titan-embed-")
	}

	/**
	 * Checks if the model is a Cohere model
	 */
	private isCohereModel(modelId: string): boolean {
		return modelId.startsWith("cohere.embed-")
	}

	/**
	 * Processes embeddings for Titan models (single text per request)
	 */
	private async processTitanEmbeddings(
		texts: string[],
		modelId: string,
	): Promise<{ embeddings: number[][]; usage: { promptTokens: number; totalTokens: number } }> {
		const embeddings: number[][] = []
		let totalTokens = 0

		// Process with controlled concurrency
		const semaphore = new Semaphore(this.config.maxConcurrency!)
		const promises = texts.map(async (text) => {
			return semaphore.acquire(async () => {
				const result = await this.embedTitanText(text, modelId)
				totalTokens += result.tokens
				return result.embedding
			})
		})

		const results = await Promise.all(promises)
		embeddings.push(...results)

		return {
			embeddings,
			usage: { promptTokens: totalTokens, totalTokens },
		}
	}

	/**
	 * Processes embeddings for Cohere models (batched)
	 */
	private async processCohereEmbeddings(
		texts: string[],
		modelId: string,
	): Promise<{ embeddings: number[][]; usage: { promptTokens: number; totalTokens: number } }> {
		const allEmbeddings: number[][] = []
		let totalTokens = 0
		const batchSize = this.config.batchSize!

		for (let i = 0; i < texts.length; i += batchSize) {
			const batch = texts.slice(i, i + batchSize)
			const result = await this.embedCohereBatch(batch, modelId)
			allEmbeddings.push(...result.embeddings)
			totalTokens += result.tokens
		}

		return {
			embeddings: allEmbeddings,
			usage: { promptTokens: totalTokens, totalTokens },
		}
	}

	/**
	 * Embeds a single text using Titan model
	 */
	private async embedTitanText(text: string, modelId: string): Promise<{ embedding: number[]; tokens: number }> {
		const input: InvokeModelCommandInput = {
			modelId,
			contentType: "application/json",
			accept: "application/json",
			body: JSON.stringify({
				inputText: text,
			}),
		}

		const result = await this.invokeModelWithRetries(input)
		const response = JSON.parse(new TextDecoder().decode(result.body))
		const embedding = response.embedding

		if (!Array.isArray(embedding)) {
			throw new Error("Invalid Titan embedding response")
		}

		// Estimate tokens (rough approximation)
		const tokens = Math.ceil(text.length / 4)
		return { embedding, tokens }
	}

	/**
	 * Embeds a batch of texts using Cohere model
	 */
	private async embedCohereBatch(
		texts: string[],
		modelId: string,
	): Promise<{ embeddings: number[][]; tokens: number }> {
		const input: InvokeModelCommandInput = {
			modelId,
			contentType: "application/json",
			accept: "application/json",
			body: JSON.stringify({
				texts,
				input_type: this.config.inputType || "search_document",
			}),
		}

		const result = await this.invokeModelWithRetries(input)
		const response = JSON.parse(new TextDecoder().decode(result.body))
		const embeddings = response.embeddings

		if (!Array.isArray(embeddings)) {
			throw new Error("Invalid Cohere embedding response")
		}

		// Estimate tokens (rough approximation)
		const totalTokens = texts.reduce((sum, text) => sum + Math.ceil(text.length / 4), 0)
		return { embeddings, tokens: totalTokens }
	}

	/**
	 * Invokes the Bedrock model with retry logic
	 */
	private async invokeModelWithRetries(input: InvokeModelCommandInput): Promise<InvokeModelCommandOutput> {
		for (let attempt = 0; attempt < this.maxRetries; attempt++) {
			try {
				const command = new InvokeModelCommand(input)
				return await this.client.send(command)
			} catch (error: any) {
				const hasMoreAttempts = attempt < this.maxRetries - 1

				// Check if it's a throttling error
				const httpError = error as HttpError
				if (httpError?.status === 429 && hasMoreAttempts) {
					const delayMs = INITIAL_RETRY_DELAY_MS * Math.pow(2, attempt)
					console.warn(
						t("embeddings:rateLimitRetry", {
							delayMs,
							attempt: attempt + 1,
							maxRetries: this.maxRetries,
						}),
					)
					await new Promise((resolve) => setTimeout(resolve, delayMs))
					continue
				}

				// Capture telemetry before reformatting the error
				TelemetryService.instance.captureEvent(TelemetryEventName.CODE_INDEX_ERROR, {
					error: error instanceof Error ? error.message : String(error),
					stack: error instanceof Error ? error.stack : undefined,
					location: "AmazonBedrockEmbeddingProvider:invokeModelWithRetries",
					attempt: attempt + 1,
				})

				// Log the error for debugging
				console.error(`Bedrock embedder error (attempt ${attempt + 1}/${this.maxRetries}):`, error)

				// Format and throw the error
				throw formatEmbeddingError(error, this.maxRetries)
			}
		}

		throw new Error(t("embeddings:failedMaxAttempts", { attempts: this.maxRetries }))
	}
}

/**
 * Simple semaphore for controlling concurrency
 */
class Semaphore {
	private permits: number
	private waiting: (() => void)[] = []

	constructor(permits: number) {
		this.permits = permits
	}

	async acquire<T>(fn: () => Promise<T>): Promise<T> {
		if (this.permits > 0) {
			this.permits--
			try {
				return await fn()
			} finally {
				this.permits++
				if (this.waiting.length > 0) {
					const resolve = this.waiting.shift()!
					resolve()
				}
			}
		} else {
			return new Promise((resolve, reject) => {
				this.waiting.push(async () => {
					this.permits--
					try {
						const result = await fn()
						resolve(result)
					} catch (error) {
						reject(error)
					} finally {
						this.permits++
						if (this.waiting.length > 0) {
							const resolve = this.waiting.shift()!
							resolve()
						}
					}
				})
			})
		}
	}
}
