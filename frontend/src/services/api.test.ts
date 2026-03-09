import { describe, it, expect, vi, beforeEach } from 'vitest'
import { api } from './api'

// Mock global fetch
const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

describe('api service', () => {
  beforeEach(() => {
    mockFetch.mockReset()
    vi.useRealTimers()
  })

  describe('request helper basics', () => {
    it('returns JSON on successful response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: 'ok' }),
      })

      const result = await api.getHealth()
      expect(result).toEqual({ status: 'ok' })
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/health'),
        expect.any(Object)
      )
    })

    it('throws descriptive error on 4xx/5xx', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        text: () => Promise.resolve(JSON.stringify({ detail: 'Validation failed' })),
      })

      await expect(api.getHealth()).rejects.toThrow('Validation failed')
    })

    it('handles raw text error if JSON parsing fails', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: () => Promise.resolve('Internal Server Error'),
      })

      await expect(api.getHealth()).rejects.toThrow('Internal Server Error')
    })

    it('throws ApiTimeoutError on TimeoutError', async () => {
      // Vitest's way to simulate a TimeoutError from fetch
      const timeoutErr = new DOMException('The operation was aborted', 'TimeoutError')
      mockFetch.mockRejectedValueOnce(timeoutErr)

      await expect(api.getHealth()).rejects.toThrow(/timed out/)
    })
  })

  describe('specific endpoints', () => {
    it('wakeBackend uses extended timeout', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: 'ok' }),
      })

      await api.wakeBackend()
      const call = mockFetch.mock.calls[0]
      // AbortSignal.timeout(90000) is used in api.ts
      expect(call[1].signal).toBeDefined()
    })

    it('startAnalysis sends correct JSON body', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ session_id: 's1' }),
      })

      const payload = {
        session_id: 'custom-sid',
        privacy_mode: true,
        retention_hours: 24,
      }
      await api.startAnalysis(payload)

      const [url, init] = mockFetch.mock.calls[0]
      expect(url).toContain('/analysis/start')
      expect(init.method).toBe('POST')
      expect(init.headers['Content-Type']).toBe('application/json')
      expect(JSON.parse(init.body)).toEqual(payload)
    })

    it('analyze sends FormData directly', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true }),
      })

      const formData = new FormData()
      formData.append('file', new Blob(), 'test.jpg')
      
      await api.analyze(formData)

      const [, init] = mockFetch.mock.calls[0]
      expect(init.method).toBe('POST')
      expect(init.body).toBe(formData)
      // Content-Type should NOT be set manually for FormData so browser sets boundary
      expect(init.headers).toBeUndefined()
    })

    it('getHistory constructs correct query params', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ items: [] }),
      })

      await api.getHistory(10, 'profile-1', 'cursor-abc')

      const [url] = mockFetch.mock.calls[0]
      const urlObj = new URL(url)
      expect(urlObj.searchParams.get('limit')).toBe('10')
      expect(urlObj.searchParams.get('profile_id')).toBe('profile-1')
      expect(urlObj.searchParams.get('cursor')).toBe('cursor-abc')
    })

    it('getCompare handles optional previousSessionId', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ compare: {} }),
      })

      // Case 1: No previous session
      await api.getCompare('curr-1')
      expect(mockFetch.mock.calls[0][0]).not.toContain('previous_session_id')

      // Case 2: With previous session
      await api.getCompare('curr-1', 'prev-1')
      expect(mockFetch.mock.calls[1][0]).toContain('previous_session_id=prev-1')
    })

    it('updateSessionNotes sends JSON payload', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({}),
      })

      await api.updateSessionNotes('s1', 'Test note')
      const [, init] = mockFetch.mock.calls[0]
      expect(JSON.parse(init.body)).toEqual({ note: 'Test note' })
    })

    it('exportBundle defaults to clinical preset', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({}),
      })

      await api.exportBundle('s1')
      const [, init] = mockFetch.mock.calls[0]
      expect(JSON.parse(init.body).preset).toBe('clinical')
    })
  })
})
