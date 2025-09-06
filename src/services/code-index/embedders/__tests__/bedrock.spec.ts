import type { MockedClass, MockedFunction } from "vitest"
import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime"

import { AmazonBedrockEmbeddingProvider } from "../bedrock"
import { INITIAL_RETRY_DELAY_MS } from "../../constants"

// Mock the AWS SDK
vitest.mock("@aws-sdk/client-bedrock-runtime", () => {
	return {
		BedrockRuntimeClient: vitest.fn(),
		InvokeModelCommand: vitest.fn(),
	}
})

// Mock TelemetryService
vitest.mock("@roo-code/telemetry", () => ({
	TelemetryService: {
		instance: {
			captureEvent: vitest.fn(),
		},
	},
}))

// Mock i18n
vitest.mock("../../../../i18n", () => ({
	t: (key: string, params?: Record<string, any>) => {
		const translations: Record<string, string> = {
			"embeddings:authenticationFailed":
				"Failed to create embeddings: Authentication failed. Please check your AWS credentials.",
			"embeddings:validation.authenticationFailed": "embeddings:validation.authenticationFailed",
			"embeddings:validation.connectionFailed": "embeddings:validation.connectionFailed",
			"embeddings:failedWithStatus": `Failed to create embeddings after ${params?.attempts} attempts: HTTP ${params?.statusCode} - ${params?.errorMessage}`,
			"embeddings:failedWithError": `Failed to create embeddings after ${params?.attempts} attempts: ${params?.errorMessage}`,
			"embeddings:failedMaxAttempts": `Failed to create embeddings after ${params?.attempts} attempts`,
			"embeddings:rateLimitRetry": `Rate limit hit, retrying in ${params?.delayMs}ms (attempt ${params?.attempt}/${params?.maxRetries})`,
			"embeddings:unsupportedModel": `Unsupported model: ${params?.model}`,
		}
		return translations[key] || key
	},
}))

// Mock console methods
const consoleMocks = {
	error: vitest.spyOn(console, "error").mockImplementation(() => {}),
	warn: vitest.spyOn(console, "warn").mockImplementation(() => {}),
}

describe("AmazonBedrockEmbeddingProvider", () => {
	let embedder: AmazonBedrockEmbeddingProvider
	let mockSend: MockedFunction<any>
	let MockedBedrockRuntimeClient: MockedClass<typeof BedrockRuntimeClient>

	beforeEach(() => {
		vitest.clearAllMocks()
		consoleMocks.error.mockClear()
		consoleMocks.warn.mockClear()

		MockedBedrockRuntimeClient = BedrockRuntimeClient as MockedClass<typeof BedrockRuntimeClient>
		mockSend = vitest.fn()

		const mockClient = {
			send: mockSend,
		}
		MockedBedrockRuntimeClient.mockReturnValue(mockClient as any)

		embedder = new AmazonBedrockEmbeddingProvider({
			modelId: "amazon.titan-embed-text-v1",
			region: "us-east-1",
		})
	})

	afterEach(() => {
		vitest.clearAllMocks()
	})

	describe("constructor", () => {
		it("should initialize with provided options", () => {
			expect(MockedBedrockRuntimeClient).toHaveBeenCalledWith({
				region: "us-east-1",
				endpoint: undefined,
				credentials: {
					accessKeyId: "",
					secretAccessKey: "",
					sessionToken: undefined,
				},
				maxAttempts: 1,
			})
			expect(embedder.embedderInfo.name).toBe("bedrock")
		})

		it("should use default values for optional parameters", () => {
			const embedderWithDefaults = new AmazonBedrockEmbeddingProvider({
				modelId: "amazon.titan-embed-text-v1",
			})

			expect(embedderWithDefaults).toBeDefined()
		})
	})

	describe("createEmbeddings", () => {
		describe("Titan models", () => {
			beforeEach(() => {
				embedder = new AmazonBedrockEmbeddingProvider({
					modelId: "amazon.titan-embed-text-v1",
					maxConcurrency: 1, // For predictable testing
				})
			})

			it("should create embeddings for a single text", async () => {
				const testTexts = ["Hello world"]
				const mockResponse = {
					body: new TextEncoder().encode(JSON.stringify({ embedding: [0.1, 0.2, 0.3] })),
				}
				mockSend.mockResolvedValue(mockResponse)

				const result = await embedder.createEmbeddings(testTexts)

				expect(mockSend).toHaveBeenCalledWith(expect.any(Object))
				expect(result).toEqual({
					embeddings: [[0.1, 0.2, 0.3]],
					usage: { promptTokens: 3, totalTokens: 3 },
				})
			})

			it("should create embeddings for multiple texts with concurrency control", async () => {
				const testTexts = ["Hello", "world"]
				const mockResponse1 = {
					body: new TextEncoder().encode(JSON.stringify({ embedding: [0.1, 0.2, 0.3] })),
				}
				const mockResponse2 = {
					body: new TextEncoder().encode(JSON.stringify({ embedding: [0.4, 0.5, 0.6] })),
				}
				mockSend.mockResolvedValueOnce(mockResponse1).mockResolvedValueOnce(mockResponse2)

				const result = await embedder.createEmbeddings(testTexts)

				expect(mockSend).toHaveBeenCalledTimes(2)
				expect(result.embeddings).toHaveLength(2)
				expect(result.usage?.promptTokens).toBe(4) // 2 tokens per text (ceil(length/4))
			})
		})
		it("should process multiple texts with concurrency limit (sequential calls)", async () => {
			// Set maxConcurrency to 1 for sequential processing
			embedder = new AmazonBedrockEmbeddingProvider({
				modelId: "amazon.titan-embed-text-v1",
				maxConcurrency: 1, // Limit to 1 concurrent call
			})

			const testTexts = ["Text1", "Text2", "Text3"] // 3 texts with concurrency=1
			const mockResponse1 = {
				body: new TextEncoder().encode(JSON.stringify({ embedding: [0.1, 0.2, 0.3] })),
			}
			const mockResponse2 = {
				body: new TextEncoder().encode(JSON.stringify({ embedding: [0.4, 0.5, 0.6] })),
			}
			const mockResponse3 = {
				body: new TextEncoder().encode(JSON.stringify({ embedding: [0.7, 0.8, 0.9] })),
			}
			mockSend
				.mockResolvedValueOnce(mockResponse1)
				.mockResolvedValueOnce(mockResponse2)
				.mockResolvedValueOnce(mockResponse3)

			const result = await embedder.createEmbeddings(testTexts)

			// Should make 3 sequential calls due to concurrency limit of 1
			expect(mockSend).toHaveBeenCalledTimes(3)
			expect(result.embeddings).toHaveLength(3)
			expect(result.embeddings).toEqual([
				[0.1, 0.2, 0.3],
				[0.4, 0.5, 0.6],
				[0.7, 0.8, 0.9],
			])
		})

		describe("Cohere models", () => {
			beforeEach(() => {
				embedder = new AmazonBedrockEmbeddingProvider({
					modelId: "cohere.embed-english-v3",
					batchSize: 2,
				})
			})

			it("should create embeddings for multiple texts in batches", async () => {
				const testTexts = ["Hello", "world", "test"]
				const mockResponse1 = {
					body: new TextEncoder().encode(
						JSON.stringify({
							embeddings: [
								[0.1, 0.2, 0.3],
								[0.4, 0.5, 0.6],
							],
						}),
					),
				}
				const mockResponse2 = {
					body: new TextEncoder().encode(JSON.stringify({ embeddings: [[0.7, 0.8, 0.9]] })),
				}
				mockSend.mockResolvedValueOnce(mockResponse1).mockResolvedValueOnce(mockResponse2)

				const result = await embedder.createEmbeddings(testTexts)

				expect(mockSend).toHaveBeenCalledTimes(2)
				expect(result.embeddings).toHaveLength(3)
			})
			it("should create embeddings for texts within single batch (explicit batching test)", async () => {
				const testTexts = ["Hello", "world"] // 2 texts, default batchSize=32 → single batch
				const mockResponse = {
					body: new TextEncoder().encode(
						JSON.stringify({
							embeddings: [
								[0.1, 0.2, 0.3],
								[0.4, 0.5, 0.6],
							],
						}),
					),
				}
				mockSend.mockResolvedValue(mockResponse)

				const result = await embedder.createEmbeddings(testTexts)

				expect(mockSend).toHaveBeenCalledTimes(1) // Single batch call for 2 texts
				expect(result.embeddings).toHaveLength(2)
				expect(result.embeddings).toEqual([
					[0.1, 0.2, 0.3],
					[0.4, 0.5, 0.6],
				])
			})
		})

		it("should use custom model when provided", async () => {
			const testTexts = ["Hello world"]
			const customModel = "cohere.embed-multilingual-v3"
			const mockResponse = {
				body: new TextEncoder().encode(JSON.stringify({ embeddings: [[0.1, 0.2, 0.3]] })),
			}
			mockSend.mockResolvedValue(mockResponse)

			await embedder.createEmbeddings(testTexts, customModel)

			expect(mockSend).toHaveBeenCalledWith(expect.any(Object))
		})

		it("should throw error for unsupported model", async () => {
			const testTexts = ["Hello world"]
			const unsupportedModel = "unsupported.model"

			await expect(embedder.createEmbeddings(testTexts, unsupportedModel)).rejects.toThrow(
				"Unsupported model: unsupported.model",
			)
		})

		describe("error handling", () => {
			it("should handle API errors gracefully", async () => {
				const testTexts = ["Hello world"]
				const apiError = new Error("API connection failed")

				mockSend.mockRejectedValue(apiError)

				await expect(embedder.createEmbeddings(testTexts)).rejects.toThrow(
					"Failed to create embeddings after 3 attempts: API connection failed",
				)

				expect(console.error).toHaveBeenCalledWith(
					expect.stringContaining("Bedrock embedder error"),
					expect.any(Error),
				)
			})

			it("should handle malformed responses", async () => {
				const testTexts = ["Hello world"]
				const malformedResponse = {
					body: new TextEncoder().encode(JSON.stringify({})),
				}
				mockSend.mockResolvedValue(malformedResponse)

				await expect(embedder.createEmbeddings(testTexts)).rejects.toThrow("Invalid Titan embedding response")
			})
		})
	})

	describe("validateConfiguration", () => {
		it("should validate successfully with valid configuration", async () => {
			const mockResponse = {
				body: new TextEncoder().encode(JSON.stringify({ embedding: [0.1, 0.2, 0.3] })),
			}
			mockSend.mockResolvedValue(mockResponse)

			const result = await embedder.validateConfiguration()

			expect(result.valid).toBe(true)
			expect(result.error).toBeUndefined()
		})

		it("should fail validation with authentication error", async () => {
			const authError = new Error("Unauthorized")
			;(authError as any).status = 401
			mockSend.mockRejectedValue(authError)

			describe("Error surfaces", () => {
				it("should handle permission error (403) with user message", async () => {
					const testTexts = ["Hello world"]
					const permissionError = {
						name: "HttpError",
						status: 403,
						message: "Forbidden - Permission denied",
					}
					mockSend.mockRejectedValue(permissionError)

					await expect(embedder.createEmbeddings(testTexts)).rejects.toThrow(
						"Failed to create embeddings after 3 attempts: Permission denied. Check your AWS IAM permissions.",
					)
				})

				it("should throw error for invalid region in constructor", () => {
					const createClientSpy = vi.spyOn(
						require("../../../api/providers/bedrock-shared"),
						"createBedrockRuntimeClient",
					)
					createClientSpy.mockImplementation(() => {
						throw new Error("Invalid region provided")
					})

					expect(() => {
						new AmazonBedrockEmbeddingProvider({
							modelId: "amazon.titan-embed-text-v1",
							region: "invalid-region",
						})
					}).toThrow("Invalid region provided")
				})

				it("should throw error for invalid endpoint URL", () => {
					const createClientSpy = vi.spyOn(
						require("../../../api/providers/bedrock-shared"),
						"createBedrockRuntimeClient",
					)
					createClientSpy.mockImplementation(() => {
						throw new Error("Invalid endpoint URL")
					})

					expect(() => {
						new AmazonBedrockEmbeddingProvider({
							modelId: "amazon.titan-embed-text-v1",
							endpointUrl: "invalid://url",
						})
					}).toThrow("Invalid endpoint URL")
				})
			})
			const result = await embedder.validateConfiguration()

			expect(result.valid).toBe(false)
			expect(result.error).toBe(
				"Failed to create embeddings: Authentication failed. Please check your AWS credentials.",
			)
		})

		it("should fail validation with connection error", async () => {
			const connectionError = new Error("ECONNREFUSED")
			mockSend.mockRejectedValue(connectionError)

			const result = await embedder.validateConfiguration()

			expect(result.valid).toBe(false)
			expect(result.error).toBe("embeddings:validation.connectionFailed")
		})
	})

	describe("retry logic", () => {
		beforeEach(() => {
			vitest.useFakeTimers()
		})

		afterEach(() => {
			vitest.useRealTimers()
		})

		it("should retry on throttling errors with exponential backoff", async () => {
			const testTexts = ["Hello world"]
			const throttleError = { status: 429, message: "Too Many Requests" }

			mockSend
				.mockRejectedValueOnce(throttleError)
				.mockRejectedValueOnce(throttleError)
				.mockResolvedValueOnce({
					body: new TextEncoder().encode(JSON.stringify({ embedding: [0.1, 0.2, 0.3] })),
				})

			const resultPromise = embedder.createEmbeddings(testTexts)

			// Fast-forward through the delays
			await vitest.advanceTimersByTimeAsync(INITIAL_RETRY_DELAY_MS)
			await vitest.advanceTimersByTimeAsync(INITIAL_RETRY_DELAY_MS * 2)

			const result = await resultPromise

			expect(mockSend).toHaveBeenCalledTimes(3)
			expect(console.warn).toHaveBeenCalledWith(expect.stringContaining("Rate limit hit, retrying in"))
			expect(result.embeddings).toHaveLength(1)
		})
	})
})
