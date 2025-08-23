# MIT License
#
# Copyright (c) 2025 Efe Özyay
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#                                BNBW      @QMg
#                                BWBW      Q@MN
#
#                              _,ef#80DD80DSh>+
#                            =?O#fC_-    :`ri&R3V
#                          "<%dv|            .`kweF
#                          3V7o              ,_2XWQ.`
#                        /+69~"          .`J}#8Y(!^
#                        znAk.`        Lx#8IF`.
#                        jC2X      <>69Kb!^
#                        {}hw  ,'2XR&Lc
#                        |cbK7oNBCf.`
#                        `.#8dp;^
#                          |cHE^!              {}IF
#                            zs%Gv|        ~!hwdp',
#                              ?*9qO#wkZyHpOG}{`:
#
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "google-genai>=1.31.0",
# ]
# ///

"""A command-line tool to automatically fetch, summarize, and display random articles.

This script is designed to keep a GitHub profile's README fresh and engaging by
periodically updating it with summaries of random articles from the SCP Foundation and
Wikipedia. It leverages the SCP Foundation's GraphQL API to fetch a random SCP entry and
a predefined list of URLs to select a random Wikipedia article. The content of these
articles is then summarized using Google's Gemini Large Language Model (LLM). The final
summaries are integrated into a README template, which can then be used to update a
repository's main page.

Features:
    - Fetches a random SCP article using the SCP Foundation's GraphQL API.
    - Selects a random Wikipedia article from a user-provided list.
    - Utilizes Google's Gemini family of models for high-quality content summarization.
    - Intelligently updates a target README file, preserving manual edits by only
      modifying auto-generated sections.
    - Robust error handling with custom exceptions and built-in retry mechanisms for API
      calls.
    - Configurable via both environment variables and command-line arguments.
    - Structured logging for clear and effective debugging.

Usage:
    The script is executed from the command line. It requires certain environment
    variables to be set for authentication and prompts.

    ```sh
    $ export GH_SUMMARIZER_GEMINI_API_KEY="your_gemini_api_key"
    $ export GH_SUMMARIZER_SCP_USER_PROMPT="Summarize the SCP article at {scp_url}."
    $ export GH_SUMMARIZER_WIKIPEDIA_USER_PROMPT="Summarize this: {wikipedia_url}."
    $ python gh_summarizer.py -o README.md
    ```

    Command-line arguments can be used to override default settings and environment
    variables. Use `python gh_summarizer.py --help` for a full list of options.

Configuration:
    The script is configured through the following environment variables:
    - `GH_SUMMARIZER_GEMINI_API_KEY`: (Required) Your API key for the Google Gemini
      service.
    - `GH_SUMMARIZER_SCP_USER_PROMPT`: (Required) The prompt template for summarizing
      SCP articles. Must include the `{scp_url}` placeholder.
    - `GH_SUMMARIZER_WIKIPEDIA_USER_PROMPT`: (Required) The prompt template for
      summarizing Wikipedia articles. Must include the `{wikipedia_url}` placeholder.
    - `GH_SUMMARIZER_SCP_SYSTEM_PROMPT`: (Optional) A system-level prompt to guide the
      LLM's behavior for SCP summaries.
    - `GH_SUMMARIZER_WIKIPEDIA_SYSTEM_PROMPT`: (Optional) A system-level prompt for
      Wikipedia summaries.

Design Philosophy & Implementation Notes:
    - Error Handling and Logging in Exceptions:
      This script places logging calls directly within the `__init__` method of its
      custom exception classes. While this is often considered an anti-pattern, it was a
      deliberate design choice for this application. The script is a command-line tool
      where these exceptions are fatal and terminate the program. By logging at
      instantiation, we ensure that any critical failure is immediately recorded without
      needing explicit logging calls in every `except` block. This simplifies the main
      application logic and guarantees that no error goes unlogged.

    - Configuration Management:
      A hybrid approach to configuration allows for flexibility. Environment variables
      are used for default and sensitive settings (like API keys), which is ideal for
      CI/CD environments (e.g., GitHub Actions). Command-line arguments provide a
      convenient way to override these settings for local testing or specific runs.

    - Modularity and Separation of Concerns:
      The script is divided into distinct functions, each with a single responsibility:
      fetching data (`get_random_scp`, `get_random_wikipedia_url`), processing data
      (`summarize`), and presenting data (`generate_readme`). This separation makes the
      code easier to read, test, and maintain.

    - Robustness and Resilience:
      Network and API interactions are inherently unreliable. The script incorporates
      retry mechanisms for API calls to handle transient failures gracefully.
      Furthermore, the `generate_readme` function's intelligent update logic ensures
      that a failure in one part of the process (e.g., summarizing a Wikipedia article)
      does not prevent the successful parts (e.g., the SCP summary) from being updated.

    - Modern Python Practices:
      The script utilizes modern Python features such as `pathlib` for path manipulation
      and type hints for improved code clarity. It also declares its dependencies using
      the PEP 723 `/// script` block, making it a self-contained and easily
      distributable tool.
"""

__version__ = "2.0.0"

import argparse
import dataclasses as dc
import difflib
import json
import logging
import os
import secrets
import sys
import typing
import urllib.parse as uparse
import urllib.request as ureq
from pathlib import Path

from google import genai as ggenai
from google.genai import types as gtypes

logger = logging.getLogger(__name__)


class MissingEnvironmentVariableError(ValueError):
    """Exception raised when a required environment variable is not set.

    This exception is raised when an environment variable that is essential for the
    application's operation is missing. It provides a detailed error message that
    includes the name of the missing variable, its purpose, and optional instructions
    for resolving the issue.

    The error message is constructed using a predefined template to ensure consistent
    and informative error reporting.

    Args:
        var_name: The name of the missing environment variable.
        purpose: A description of why the variable is needed.
        additional_info: Any extra information or instructions for the user. Defaults to
            an empty string.

    Attributes:
        MSG_TEMPLATE: A format string template for constructing the error message. It
            uses `{var_name}`, `{purpose}`, and `{additional_info}` as placeholders.

    Example:
        >>> import os
        >>> if (api_key := os.environ.get("API_KEY")) is None:
        ...     raise MissingEnvironmentVariableError(
        ...         "API_KEY",
        ...         "API authentication",
        ...         "Get one from: https://api.example.com/keys",
        ...     )
    """

    MSG_TEMPLATE = (
        "Required environment variable `{var_name}` is not set. "
        "This variable is necessary for {purpose}. "
        "{additional_info}"
    )

    def __init__(self, var_name: str, purpose: str, additional_info: str = "") -> None:
        """Initializes the `MissingEnvironmentVariableError` exception.

        This constructor formats the error message using the `MSG_TEMPLATE` and the
        provided arguments. It then calls the parent `ValueError` constructor with the
        formatted message and logs the error as a critical issue.
        """
        error_message = self.MSG_TEMPLATE.format(
            var_name=var_name,
            purpose=purpose,
            additional_info=additional_info,
        )

        super().__init__(error_message)
        logger.critical(error_message)


class LLMResponseError(Exception):
    """Base exception for Large Language Model (LLM) response errors.

    This exception is raised for issues during interaction with an LLM, such as
    unexpected responses, invalid content, or other processing failures. It formats a
    standardized error message and logs it for debugging.

    Args:
        resp_text: The problematic response from the LLM. This can be the raw response
            text or a dictionary with error details from the LLM API.

    Attributes:
        MSG_TEMPLATE: A format string for the error message. It includes a `{resp_text}`
            placeholder for the raw LLM response or error details.

    Example:
        >>> try:
        ...     # The `summarize` function internally checks for and raises this error.
        ...     summary = summarize(client, "Bad URL")
        ... except LLMResponseError as e:
        ...     print(f"LLM Error occurred: {e}")
        ...     # Handle general LLM errors, perhaps by using a fallback system.
    """

    MSG_TEMPLATE = (
        "An unexpected error occurred during the language model interaction. "
        "Error details: {resp_text}. This may indicate a temporary service disruption "
        "or an issue with the model's processing of your request."
    )

    def __init__(self, resp_text: str | dict[typing.Any, typing.Any]) -> None:
        """Initializes the `LLMResponseError` exception.

        This constructor formats an error message using the `MSG_TEMPLATE` and the
        provided response text. It then calls the parent `Exception` constructor and
        logs the formatted message as an error.
        """
        error_message = self.MSG_TEMPLATE.format(resp_text=resp_text)

        super().__init__(error_message)
        logger.error(error_message)


class LLMResponseInvalidError(LLMResponseError):
    """Exception for invalid, error-containing, or unprocessable LLM responses.

    This exception is raised when the Large Language Model (LLM) returns a response that
    is invalid or unusable. This can occur due to network issues, an unreachable URL,
    malformed or nonsensical model output, or API service interruptions.

    The exception uses a predefined message template that includes the invalid response
    text and possible failure reasons.

    Args:
        resp_text: The problematic LLM response, which can be either the raw response
            text or a dictionary with error details from the LLM API.

    Attributes:
        MSG_TEMPLATE: A format string for the error message, including placeholders for
            the response text and explanatory details.

    Example:
        >>> try:
        ...     # The `summarize` function internally checks for and raises this error.
        ...     summary = summarize(client, "Invalid URL")
        ... except LLMResponseInvalidError as e:
        ...     print(f"Failed to generate summary: {e}")
        ...     # Handle invalid response, perhaps by retrying or using a fallback.

    Note:
        This exception is typically caught and handled in the summarization workflow to
        implement retry logic or fallback mechanisms when the LLM fails to generate
        valid content.
    """

    MSG_TEMPLATE = (
        "The language model returned an invalid response: {resp_text}. "
        "This could be due to several reasons: the URL may be unreachable, "
        "there might be network connectivity issues, or the model may have "
        "generated inaccurate or nonsensical content."
    )


class LLMResponseInappropriateError(LLMResponseError):
    """Exception for LLM responses containing inappropriate content.

    This exception handles cases where the Large Language Model (LLM) generates content
    that is flagged as inappropriate or unsuitable.

    This can occur due to inappropriate source material, model hallucinations, or a
    response that violates content safety policies by being harmful, containing
    sensitive information, or failing automated checks.

    The exception helps maintain content safety standards by allowing the application to
    gracefully handle and report inappropriate content generation, enabling proper error
    handling and logging of such incidents.

    Args:
        resp_text: The problematic response from the LLM. This can be the raw response
            text or a dictionary with error details from the LLM API.

    Attributes:
        MSG_TEMPLATE: A format string template for constructing error messages. Contains
            a placeholder for `resp_text` which includes details about the inappropriate
            content and possible causes.

    Example:
        >>> try:
        ...     # The `summarize` function internally checks for and raises this error.
        ...     summary = summarize(client, "http://example.com/unsafe-content")
        ... except `LLMResponseInappropriateError` as e:
        ...     logger.warning(f"Content safety violation detected: {e}")
        ...     # Handle inappropriate content, perhaps by using a fallback or notifying
        ...     # the user.

    Note:
        This exception is part of the content safety system and works in conjunction
        with the LLM's built-in content filtering. It provides an additional layer of
        protection against inappropriate content generation and helps maintain the
        quality and safety of the summarization service.
    """

    MSG_TEMPLATE = (
        "The language model generated inappropriate content: {resp_text}. "
        "This could be due to several reasons: the source URL may contain "
        "inappropriate material (such as adult content, hate speech, or graphic "
        "violence), or the model may have generated unsuitable content despite safe "
        "input."
    )


class LLMResponseTooShortError(LLMResponseError):
    """Exception for LLM responses that are too short to be valid summaries.

    This exception is raised when a generated summary from the Large Language Model
    (LLM) falls below a minimum length threshold. It serves as a quality control
    mechanism to prevent inadequate or incomplete summaries from being used.

    This can be caused by sparse source content, model processing failures, API or
    network issues, rate limiting, or the model determining there is not enough
    meaningful content to summarize.

    Args:
        resp_text: The problematic response from the LLM. This can be the raw response
            text that was too short or a dictionary with detailed error information from
            the LLM API.

    Attributes:
        MSG_TEMPLATE: A format string template for constructing error messages. Contains
            a placeholder for `resp_text` which includes the short response and possible
            reasons for the brevity.

    Example:
        >>> try:
        ...     summary = summarize(client, url)
        ...     if len(summary) <= 200:
        ...         raise LLMResponseTooShortError(summary)
        ... except LLMResponseTooShortError as e:
        ...     logger.warning(f"Summary too short: {e}")
        ...     # Handle the short response, perhaps by requesting a new summary with
        ...     # different parameters or from a different model.
    """

    MSG_TEMPLATE = (
        "The generated summary is too short: {resp_text}. "
        "This could be due to several factors: the source content may be too sparse, "
        "the model may have failed to properly understand the content, or there might "
        "be API rate limiting affecting response quality."
    )


class SCPResponseInvalidError(Exception):
    """Exception for invalid or error responses from the SCP Foundation's GraphQL API.

    This exception handles error responses from the SCP Foundation GraphQL API. It
    captures failures like network issues, API downtime, malformed responses, and server
    errors.

    The exception uses a predefined message template that includes error details and
    HTTP status for context.

    Args:
        resp_text: The error response text received from the API. This could be raw text
            or a JSON-formatted error message.
        status_code: The HTTP status code returned by the API request. Common codes
            include 400 (Bad Request), 401 (Unauthorized), 403 (Forbidden),
            404 (Not Found), and 500 (Internal Server Error).

    Attributes:
        MSG_TEMPLATE: A format string template for constructing error messages. Contains
            placeholders for response text and HTTP status code.

    Example:
        >>> try:
        ...     # The `get_random_scp` function internally checks for and raises this
        ...     # error.
        ...     response = get_random_scp()
        ... except SCPResponseInvalidError as e:
        ...     print(f"API request failed: {e}")
        ...     # Handle API errors, perhaps by implementing retry logic or falling back
        ...     # to cached content

    Note:
        This exception is typically caught in the main execution flow to implement retry
        mechanisms or fallback strategies when API requests fail. It's important to log
        these failures for monitoring API health.
    """

    MSG_TEMPLATE = (
        "Unable to retrieve data from the SCP Foundation GraphQL API. "
        "Response returned: {resp_text} (Status code: {status_code}). "
        "This could be due to the API endpoint is currently unreachable, there are "
        "network connectivity issues, or the request may have been rate-limited."
    )

    def __init__(self, resp_text: str, status_code: int) -> None:
        """Initializes the `SCPResponseInvalidError` exception.

        This constructor formats an error message using the `MSG_TEMPLATE` and the
        provided arguments. It then calls the parent `Exception` constructor and logs
        the formatted message as an error.
        """
        error_message = self.MSG_TEMPLATE.format(
            resp_text=resp_text,
            status_code=status_code,
        )
        super().__init__(error_message)
        logger.error(error_message)


@dc.dataclass(order=True, frozen=True, kw_only=True, slots=True)
class SCPData:
    """A frozen dataclass representing structured data about an SCP Foundation article.

    This immutable dataclass encapsulates essential information about an SCP Foundation
    article, including its title, alternative title, URL, and a summary.

    The [SCP Foundation](https://scp-wiki.wikidot.com/) is a collaborative writing
    project focused on cataloging and containing anomalous objects, entities, and
    phenomena. Each SCP article describes one such anomaly and its containment
    procedures.

    Attributes:
        title: The official designation and title of the SCP article as shown on the SCP
            wiki (e.g., `"SCP-173"`).
        title_alt: The alternative title or nickname for the SCP, if one exists
            (e.g., `"The Sculpture"`). This field may be empty.
        url: The complete URL to the SCP article on the SCP Foundation wiki
            (e.g., `"https://scp-wiki.wikidot.com/scp-173"`).
        summary: A concise summary of the SCP article's content, typically generated by
            a Large Language Model. The summary should capture the key aspects of the
            SCP, including its description, properties, and containment procedures.

    Examples:
        Creating an instance of `SCPData`:

        >>> article = SCPData(
        ...     title="SCP-173",
        ...     title_alt="The Sculpture",
        ...     url="https://scp-wiki.wikidot.com/scp-173",
        ...     summary="SCP-173 is a constructed concrete humanoid...",
        ... )
        >>> print(article.title)
        'SCP-173'
    """

    title: str
    title_alt: str
    url: str
    summary: str


@dc.dataclass(order=True, frozen=True, kw_only=True, slots=True)
class WikipediaData:
    """A frozen dataclass representing structured data about a Wikipedia article.

    This immutable dataclass stores essential information about a Wikipedia article,
    including its title, URL, and a generated summary.

    Attributes:
        title: The title of the Wikipedia article, typically derived from the URL by
            replacing underscores with spaces and URL-decoding the string. Example:
            `"Albert Einstein"` (from `"Albert_Einstein"`).
        url: The full URL to the Wikipedia article. This should be a valid Wikipedia URL
            following the pattern: `"https://en.wikipedia.org/wiki/[Article_Title]"`.
            Example: `"https://en.wikipedia.org/wiki/Albert_Einstein"`.
        summary: A concise summary of the Wikipedia article's content, typically
            generated by a Large Language Model. The summary should capture the key
            points while maintaining readability and accuracy.

    Note:
        This class is a frozen `dataclass` with `__slots__` enabled for memory
        efficiency and immutability. All fields are keyword-only to prevent parameter
        order confusion, and comparison methods are automatically generated.

    Examples:
        Creating an instance of `WikipediaData`:

        >>> article = WikipediaData(
        ...     title="Albert Einstein",
        ...     url="https://en.wikipedia.org/wiki/Albert_Einstein",
        ...     summary="Albert Einstein was a German-born theoretical physicist...",
        ... )
        >>> print(article.title)
        'Albert Einstein'
    """

    title: str
    url: str
    summary: str


def get_random_scp(
    max_retries: int = 3,
    graphql_api_endpoint: str = "https://api.crom.avn.sh/graphql",
) -> dict[
    str,
    dict[str, typing.Any],
]:
    """Fetches a random SCP article from the SCP Foundation's GraphQL API.

    This function queries the GraphQL endpoint to retrieve a random SCP article. It
    includes a built-in retry mechanism to handle transient network or API issues. The
    function sends a POST request with a specific GraphQL query to get article metadata,
    such as title, URL, and alternative titles.

    Args:
        max_retries: The maximum number of retry attempts for a failed request. The
            function uses a recursive approach for retries. Defaults to 3.
        graphql_api_endpoint: The URL for the GraphQL API. Must start with `http:` or
            `https:`. Defaults to `https://api.crom.avn.sh/graphql`.

    Returns:
        A nested dictionary containing the API response. The structure typically is:
            ```graphql
            {
                'data': {
                    'randomPage': {
                        'page': {
                            'alternateTitles': [{'title': str}],
                            'url': str,
                            'wikidotInfo': {'title': str},
                        }
                    }
                }
            }
            ```

    Raises:
        ValueError: If `graphql_api_endpoint` has an invalid format.
        SCPResponseInvalidError: If the API returns a non-200 status code after all
            retry attempts are exhausted.
        urllib.error.URLError: If there's a network issue, like the endpoint being
            unreachable.
        json.JSONDecodeError: If the API response is not valid JSON.

    Example:
        >>> try:
        ...     scp_data = get_random_scp()
        ...     page_info = scp_data["data"]["randomPage"]["page"]
        ...     print(f"Fetched SCP: {page_info['wikidotInfo']['title']}")
        ... except SCPResponseInvalidError as e:
        ...     print(f"Failed to fetch SCP: {e}")

    Notes:
        - We suppress Ruff rule
          [S310](https://docs.astral.sh/ruff/rules/suspicious-url-open-usage/) because
          although we explicitly check the provided endpoint for valid protocols, Ruff
          is not smart enough to understand it yet. We can stop suppressing when
          [this](https://github.com/astral-sh/ruff/issues/7918) issue is resolved.
    """
    # Validate that the API endpoint URL starts with a valid protocol
    if not graphql_api_endpoint.startswith(("http:", "https:")):
        error_message = (
            "Invalid API endpoint URL format. "
            "The SCP Foundation GraphQL API endpoint must start with `http:` or "
            f"`https:`. Provided URL: {graphql_api_endpoint}. "
            "Please provide a valid URL including the protocol."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Construct the GraphQL query to fetch a random SCP article
    # The query filters for pages on the official SCP wiki with the "scp" tag
    graphql_query = json.dumps(
        {
            "query": (
                '{randomPage(filter: {anyBaseUrl: "http://scp-wiki.wikidot.com", '
                'allTags: "scp"}) {page {alternateTitles {title}, url, '
                "wikidotInfo{title}}}}"
            ),
        },
    ).encode("UTF-8")

    # The User-Agent is set to a common browser string to avoid being blocked
    logger.info("Requesting random SCP article from SCP Foundation GraphQL API...")
    req = ureq.Request(  # noqa: S310
        url=graphql_api_endpoint,
        data=graphql_query,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.3",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Content-Length": str(len(graphql_query)),
        },
        method="POST",
    )

    resp = ureq.urlopen(req)  # noqa: S310

    logger.info("SCP Foundation GraphQL API request finalized!")
    logger.debug(
        "SCP Foundation GraphQL API response status: %s (%d)",
        resp.reason,
        resp.status,
    )

    resp_text = resp.read().decode("UTF-8")
    resp_json = json.loads(resp_text)

    # Handle non-200 responses by retrying until success or max retries are exhausted
    # This helps in gracefully handling temporary API issues and rate limiting
    if resp.status != 200:  # noqa: PLR2004
        if not max_retries:
            # If all retries are exhausted, raise an exception
            raise SCPResponseInvalidError(resp_text, resp.status)

        max_retries -= 1

        logger.warning(
            "Retrying SCP Foundation GraphQL API request, %d attempts remained...",
            max_retries,
        )
        # Recursively call the function to retry the request
        resp_json = get_random_scp(max_retries)

    logger.info("SCP Foundation GraphQL API response is successful: %s", resp_json)
    return resp_json


def get_random_wikipedia_url(wikipedia_urls_fp: str = "./wikipedia_urls.txt") -> str:
    """Retrieves a random Wikipedia article URL from a provided text file.

    This function reads a file of line-separated Wikipedia URLs and returns one at
    random. It uses `secrets.choice` for cryptographically secure selection, ensuring a
    high degree of randomness. The function is designed for robust file handling, using
    `pathlib.Path` and enforcing UTF-8 encoding.

    Args:
        wikipedia_urls_fp: The path to a UTF-8 encoded text file containing one complete
            Wikipedia URL per line. The file should not contain empty lines or comments.
            Defaults to `./wikipedia_urls.txt`.

    Returns:
        A randomly selected Wikipedia article URL, stripped of leading/trailing
        whitespace.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If the process lacks permission to read the file.
        IOError: If there are other problems reading the file.
        UnicodeDecodeError: If the file is not properly UTF-8 encoded.
        IndexError: If the file is empty and `secrets.choice` is called on an empty
            sequence.

    Example:
        >>> try:
        ...     url = get_random_wikipedia_url("./my_wikipedia_urls.txt")
        ...     print(f"Random article URL: {url}")
        ... except FileNotFoundError:
        ...     print("URLs file not found!")
        ... except IndexError:
        ...     print("URLs file is empty!")

    Note:
        For production use, ensure the input file exists, is properly formatted, and
        contains at least one URL to avoid exceptions.
    """
    wikipedia_urls_path = Path(wikipedia_urls_fp)

    logger.info(
        "Getting a random Wikipedia article URL from %s...",
        wikipedia_urls_path,
    )
    wikipedia_urls = wikipedia_urls_path.read_text(encoding="UTF-8").splitlines()

    wikipedia_url = secrets.choice(wikipedia_urls).strip()
    logger.info("Got random Wikipedia article URL: %s", wikipedia_url)

    return wikipedia_url


def summarize(
    gemini_client: ggenai.Client,
    user_prompt: str,
    system_prompt: str | None = None,
    model: str = "gemini-2.5-flash-lite",
    max_retries: int = 3,
) -> str:
    """Generates a summary of content using Google's Gemini Large Language Model.

    This function interfaces with the Gemini API to generate summaries from a given URL.
    It implements a recursive retry mechanism for transient failures and validates the
    response quality through content safety and length checks.

    The generation process is optimized for summarization tasks by using a temperature
    of 0.6 for a balance of creativity and consistency, and an unlimited thinking budget
    for thorough content analysis.

    Args:
        gemini_client: An initialized Google Generative AI client instance with valid
            API credentials. Must have appropriate permissions for content generation.
        user_prompt: The prompt to send to the model. This typically includes
            instructions and the content to be summarized.
        system_prompt: System-level instructions to guide the model's behavior. These
            act as meta-instructions that shape how the model approaches the user
            prompt. Defaults to `None`.
        model: The specific Gemini model to use for generation. Defaults to
            `gemini-2.5-flash-lite` which offers a good balance of speed and quality.
        max_retries: Maximum number of retry attempts for failed generations. Each retry
            decrements this counter. When it reaches 0, the function will raise an
            exception instead of retrying. Defaults to 3.

    Returns:
        The generated summary, stripped of leading/trailing whitespace. The summary is
        guaranteed to be at least 200 characters long to ensure meaningful content.

    Raises:
        LLMResponseError: Base exception for all LLM-related errors.
        LLMResponseInappropriateError: When the model generates inappropriate content.
        LLMResponseInvalidError: When the model fails to generate valid content.
        LLMResponseTooShortError: When the generated summary is too short.

    Example:
        >>> from google import genai
        >>> client = genai.Client(api_key="your-api-key")
        >>> try:
        ...     summary = summarize(
        ...         client,
        ...         "Summarize this article: {url}".format(
        ...             url="https://example.com/article"
        ...         ),
        ...         system_prompt="You are a helpful summarizer.",
        ...     )
        ...     print(f"Generated summary: {summary}")
        ... except LLMResponseError as e:
        ...     print(f"Failed to generate summary: {e}")
    """
    logger.debug(
        "Summarizing with parameters:\n\t- USER PROMPT: %s\n\n\t- SYSTEM PROMPT: %s"
        "\n\n\t- MODEL: %s\n\n\t- MAX RETRIES: %s",
        user_prompt,
        system_prompt,
        model,
        max_retries,
    )

    logger.info("Requesting summarization from %s...", model)
    resp = gemini_client.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
        model=model,
        contents=user_prompt,
        config=gtypes.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.6,
            response_mime_type="text/plain",
            # Set safety thresholds to allow all content categories
            # This is required for unrestricted language processing
            safety_settings=[
                gtypes.SafetySetting(
                    category=gtypes.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=gtypes.HarmBlockThreshold.BLOCK_NONE,
                ),
                gtypes.SafetySetting(
                    category=gtypes.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=gtypes.HarmBlockThreshold.BLOCK_NONE,
                ),
                gtypes.SafetySetting(
                    category=gtypes.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=gtypes.HarmBlockThreshold.BLOCK_NONE,
                ),
                gtypes.SafetySetting(
                    category=gtypes.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=gtypes.HarmBlockThreshold.BLOCK_NONE,
                ),
            ],
            # URL Context tool is needed since we're providing the model only the URL of
            # the content we want and not the text content. The model itself extracts
            # the textual content from the web page.
            tools=[gtypes.Tool(url_context=gtypes.UrlContext())],
            thinking_config=gtypes.ThinkingConfig(thinking_budget=-1),
        ),
    )

    logger.info("Summarization request finalized!")

    if (resp_text := resp.text) is None:
        raise LLMResponseError(resp.to_json_dict())

    if "INAPPROPRIATE" in resp_text:
        logger.debug("Inappropriate response: %s", resp.to_json_dict())
        raise LLMResponseInappropriateError(resp_text)

    if "ERROR" in resp_text:
        logger.debug("Invalid response: %s", resp.to_json_dict())

        if not max_retries:
            # If all retries are exhausted, raise an exception.
            raise LLMResponseInvalidError(resp_text)

        max_retries -= 1

        # Recursively call the function to retry the summarization.
        logger.warning("Retrying summarization, %d attempts remained...", max_retries)
        resp_text = summarize(
            gemini_client,
            user_prompt,
            system_prompt,
            model,
            max_retries,
        )

    resp_text = resp_text.strip()

    # The 200-character threshold is a heuristic value determined through testing and
    # observation. It represents a balance between allowing concise summaries while
    # ensuring enough detail is provided to be meaningful.
    if len(resp_text) <= 200:  # noqa: PLR2004
        logger.debug("Too short response: %s", resp.to_json_dict())
        raise LLMResponseTooShortError(resp_text)

    resp_text = resp_text.replace("\n", " ")

    logger.info("Summarization is successful: %s", resp_text)
    return resp_text


def generate_readme(
    scp_data: SCPData | None,
    wikipedia_data: WikipediaData | None,
    readme_fp: str | None = None,
    readme_template_fp: str = "./README_TEMPLATE.md",
) -> str:
    """Generates a README file by populating a template with SCP and Wikipedia data.

    This function acts as a simple template engine, replacing placeholders in a template
    file with data from the provided `SCPData` and `WikipediaData` objects. If an output
    file path is given, it intelligently updates only the changed lines to preserve any
    manual edits.

    The ability to only update the changed lines also allows us to mitigate the
    LLM-related errors. For example, if the LLM fails to generate a valid summary for a
    Wikipedia article but the SCP summary is valid, we can just overwrite the SCP
    summary and leave the old Wikipedia one in-place.

    The template file should be a Markdown file with the following placeholders:
    - `{{SCP_URL}}`: URL to the SCP article.
    - `{{SCP_TITLE}}`: Official title of the SCP (e.g., "SCP-173").
    - `{{SCP_TITLE_ALT}}`: Alternative title (e.g., "The Sculpture").
    - `{{SCP_SUMMARY}}`: A summary of the SCP article.
    - `{{WIKIPEDIA_URL}}`: URL to the Wikipedia article.
    - `{{WIKIPEDIA_TITLE}}`: Title of the Wikipedia article.
    - `{{WIKIPEDIA_SUMMARY}}`: A summary of the Wikipedia article.

    Args:
        scp_data: A dataclass with SCP article information. If `None`, SCP-related
            placeholders are not replaced.
        wikipedia_data: A dataclass with Wikipedia article information. If `None`,
            Wikipedia-related placeholders are not replaced.
        readme_fp: The path to the output README file. If provided, the function updates
            the file in place. If `None`, it returns the generated content as a string
            without writing to a file. Defaults to `None`.
        readme_template_fp: The path to the README template file. Defaults to
            `./README_TEMPLATE.md`.

    Returns:
        The generated README content as a string.

    Raises:
        FileNotFoundError: If the template or output README file does not exist.
        PermissionError: If the process lacks permissions to read or write files.
        IOError: If any other I/O error occurs.
        UnicodeDecodeError: If a file is not UTF-8 encoded.

    Example:
        >>> # Create example data objects
        >>> scp = SCPData(
        ...     title="SCP-173",
        ...     title_alt="The Sculpture",
        ...     url="https://scp-wiki.wikidot.com/scp-173",
        ...     summary="A dangerous statue that moves when not observed...",
        ... )
        >>> wiki = WikipediaData(
        ...     title="Quantum Mechanics",
        ...     url="https://en.wikipedia.org/wiki/Quantum_mechanics",
        ...     summary="A fundamental theory in physics that describes...",
        ... )
        >>>
        >>> # Generate README with both SCP and Wikipedia data
        >>> readme = generate_readme(
        ...     scp_data=scp, wikipedia_data=wiki, readme_fp="./README.md"
        ... )
        >>> print(f"Generated README with {len(readme)} characters")
        >>>
        >>> # Generate README with only Wikipedia data
        >>> readme = generate_readme(
        ...     scp_data=None, wikipedia_data=wiki, readme_fp="./README.md"
        ... )
        >>> print(f"Updated README with Wikipedia data only")
        >>>
        >>> # Just read existing README without changes
        >>> readme = generate_readme(
        ...     scp_data=None, wikipedia_data=None, readme_fp="./README.md"
        ... )
        >>> print(f"Retrieved existing README content")

    Note:
        The in-place update mechanism for an existing README relies on line numbers and
        will only work correctly if changes do not add or remove lines from the file.
    """
    logger.info("Generating README...")

    # Read the template file content
    template_path = Path(readme_template_fp)
    readme_template = template_path.read_text(encoding="UTF-8")
    logger.debug("README template loaded: %s", readme_template)

    # If no data is provided, return the existing README or the template content
    if scp_data is None and wikipedia_data is None:
        logger.warning("No data provided for README generation!")

        if readme_fp:
            return Path(readme_fp).read_text(encoding="UTF-8")
        return readme_template

    # Create an intermediate version for template processing
    readme_inter = readme_template

    # This looks ugly, I know, but it works perfectly and it's pretty much the cleanest
    # way to do this.

    # Replace SCP-related placeholders if data is provided
    if scp_data is not None:
        logger.debug("Processing SCP data...")

        # Process each SCP field, replacing its placeholder in the template
        readme_inter = readme_inter.replace("{{SCP_URL}}", scp_data.url)
        readme_inter = readme_inter.replace("{{SCP_TITLE}}", scp_data.title)
        readme_inter = readme_inter.replace("{{SCP_TITLE_ALT}}", scp_data.title_alt)
        readme_inter = readme_inter.replace("{{SCP_SUMMARY}}", scp_data.summary)

        logger.debug("SCP data successfully processed!")

    # Replace Wikipedia-related placeholders if data is provided
    if wikipedia_data is not None:
        logger.debug("Processing Wikipedia data...")

        # Process each Wikipedia field, replacing its placeholder in the template
        readme_inter = readme_inter.replace("{{WIKIPEDIA_URL}}", wikipedia_data.url)
        readme_inter = readme_inter.replace("{{WIKIPEDIA_TITLE}}", wikipedia_data.title)
        readme_inter = readme_inter.replace(
            "{{WIKIPEDIA_SUMMARY}}",
            wikipedia_data.summary,
        )

        logger.debug("Wikipedia data successfully processed!")

    # If an output README file is provided, update the file
    if readme_fp is not None:
        logger.info("Updating output README file (%s)...", readme_fp)

        readme_path = Path(readme_fp)
        readme_text = readme_path.read_text(encoding="UTF-8")

        readme_lines = readme_text.splitlines(keepends=True)
        readme_template_lines = readme_template.splitlines(keepends=True)
        readme_inter_lines = readme_inter.splitlines(keepends=True)

        # Use `difflib` to find lines that have changed from the template
        # This allows updating only the auto-generated parts of the README
        for line in difflib.ndiff(readme_template_lines, readme_inter_lines):
            # A line starting with "+ " indicates an addition or modification
            if line.startswith("+ "):
                changed_line = line.removeprefix("+ ")
                logger.debug("Changed line: %s", changed_line)

                try:
                    # Find the index of the changed line in the intermediate
                    # representation and update the corresponding line in the output
                    # README content
                    idx = readme_inter_lines.index(changed_line)
                    readme_lines[idx] = changed_line
                except (ValueError, IndexError, Exception) as exc:
                    # This can happen if a line is completely new or line count of given
                    # output README file and the generated intermediate representation
                    # are different.
                    #
                    # A more robust diff/patch logic might be needed for complex cases
                    # but it should work fine with out use case for now.
                    error_message = (
                        f"Could not update given README file ({readme_fp})! "
                        "This could be because line counts differ between the given "
                        f"README file ({readme_fp}) and template README file "
                        f"({readme_template_fp}) or given summaries include newlines."
                    )
                    logger.exception(error_message)
                    raise ValueError(error_message) from exc

        readme_text = "".join(readme_lines)
        readme_path.write_text(readme_text, encoding="UTF-8")

        logger.info(
            "Successfully updated output README file (%s): wrote %s bytes",
            readme_fp,
            len(readme_text),
        )

        return readme_text

    return readme_inter


def _setup_logging(
    *,
    debug: bool = False,
    raise_exceptions: bool = False,
) -> logging.Logger:
    """Set up and configure a logger with multiple handlers and formatters.

    This function creates a logger that outputs log messages to:
        - Standard output (stdout) for `INFO` and `WARNING` levels.
        - Standard error (stderr) for `ERROR` and `CRITICAL` levels.

    Handlers use different formatters for concise or detailed output.

    Args:
        debug (bool, optional): If `True`, outputs debug messages to standard output.
            Defaults to `False`.
        raise_exceptions (bool, optional): If `True`, errors encountered during logging
            will raise exceptions. Set to `False` in production to suppress logging
            errors. Defaults to `False`.

    Returns:
        logging.Logger: The configured logger instance.

    Examples:
        >>> import os
        >>> logger = setup_logging()
        >>> logger.debug("Hello, world!")
        >>> logger.error("Something went wrong")

    Raises:
        No exceptions are raised by this function itself, but if `raise_exceptions` is
        `True`, errors during logging configuration may propagate.
    """
    # Determine the base logging level
    base_log_level = logging.DEBUG if debug else logging.INFO

    # Enable warning capture from the `warnings` module
    logging.captureWarnings(True)

    # Enable error handling within the logging system. This is useful for capturing
    # exceptions that occur during logging setup or configuration. Set to `False` in
    # production because we want to avoid logging errors that occur during logging.
    logging.raiseExceptions = raise_exceptions

    # Define formatters for different logging outputs
    # Basic formatter for normal stream output (minimal info)
    formatter_stream_basic = logging.Formatter(
        "%(asctime)s %(levelname)-8s # %(message)s",
        datefmt="%H:%M:%S",
    )
    # Detailed formatter for error stream (includes source location)
    formatter_stream_detailed = logging.Formatter(
        "%(asctime)s %(levelname)-8s @ (%(name)s:%(module)s:%(funcName)s:%(lineno)d) "
        "# %(message)s",
        datefmt="%H:%M:%S",
    )

    # Initialize handlers for different output destinations
    stream_handler_normal = logging.StreamHandler(stream=sys.stdout)
    stream_handler_error = logging.StreamHandler(stream=sys.stderr)

    # Configure minimum logging levels for each handler
    stream_handler_normal.setLevel(base_log_level)
    stream_handler_error.setLevel(logging.ERROR)

    # Assign formatters to handlers
    stream_handler_normal.setFormatter(formatter_stream_basic)
    stream_handler_error.setFormatter(formatter_stream_detailed)

    # Add filter to normal stream handler to exclude ERROR and CRITICAL
    stream_handler_normal.addFilter(lambda record: record.levelno <= logging.WARNING)

    # Create logger instance for this module
    logger = logging.getLogger(__name__)

    # Attach all handlers to the logger
    logger.addHandler(stream_handler_normal)
    logger.addHandler(stream_handler_error)

    # Set the base logging level (allows all messages through)
    logger.setLevel(base_log_level)

    return logger


def _main(argv: list[str] | None = None) -> None:  # noqa: C901, PLR0912, PLR0915
    """Runs the main summarization and README generation process.

    This function orchestrates the entire workflow of the summarizer script. It
    initializes the application, parses command-line arguments, validates required
    environment variables, and then proceeds to fetch, summarize, and integrate content
    from the SCP Foundation and Wikipedia into a README file.

    The process includes:
    1. Setting up the argument parser to handle user-provided configurations.
    2. Validating essential API keys and prompts from environment variables or
       command-line arguments.
    3. Initializing the Google Gemini client.
    4. Entering a loop to fetch a random SCP article, summarize it, and handle potential
        errors with a retry mechanism.
    5. Entering a similar loop for a random Wikipedia article.
    6. Generating or updating the README file with the new content.
    7. Printing the final README content to standard output.

    Args:
        argv: A list of command-line arguments to parse. If `None`, arguments are taken
            from `sys.argv`.

    Raises:
        MissingEnvironmentVariableError: If required environment variables for API keys
            or prompts are not set and not provided via command-line arguments.

    Note:
        This function is intended for internal use when the script is executed
        directly and is not part of the public API.
    """
    # Initialize environment variables. These can be overridden by command-line
    # arguments later.
    scp_system_prompt = os.environ.get("GH_SUMMARIZER_SCP_SYSTEM_PROMPT")
    wikipedia_system_prompt = os.environ.get("GH_SUMMARIZER_WIKIPEDIA_SYSTEM_PROMPT")

    # Set up the argument parser to handle command-line configurations
    argparser = argparse.ArgumentParser(
        description=(
            "A tool to fetch random articles from the SCP Foundation and Wikipedia, "
            "summarize them using Google's Gemini model, and update a README file with "
            "the results."
        ),
        epilog=(
            "This script relies on environment variables for configuration by default, "
            "but command-line arguments can be used to override them. For more "
            "information, read the project's source code. If you want to see this in "
            "action, you can visit: https://github.com/ozefe"
        ),
    )

    # --- General Arguments ---
    argparser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    argparser.add_argument(
        "--max-retries",
        "-r",
        action="store",
        default=3,
        type=int,
        help=(
            "Maximum number of retries for failed API requests. Defaults to "
            "%(default)s."
        ),
        metavar="<MAX_RETRIES>",
    )
    argparser.add_argument(
        "--model",
        "-m",
        action="store",
        default="gemini-2.5-flash-lite",
        help="The Gemini model to use for summarization. Defaults to %(default)s.",
        metavar="<GEMINI_MODEL>",
    )

    # --- File Path Arguments ---
    arg_file_ops_group = argparser.add_argument_group(
        "File operations",
        "These arguments can be used to define various input/output files.",
    )
    arg_file_ops_group.add_argument(
        "--wikipedia-urls",
        "-w",
        action="store",
        default="./wikipedia_urls.txt",
        help="Path to the Wikipedia URLs file. Defaults to %(default)s.",
        metavar="<WIKIPEDIA_URLS_FILE>",
    )
    arg_file_ops_group.add_argument(
        "--template",
        "-t",
        action="store",
        default="./README_TEMPLATE.md",
        help="Path to the README template file. Defaults to %(default)s.",
        metavar="<README_TEMPLATE_FILE>",
    )
    arg_file_ops_group.add_argument(
        "--output-readme",
        "-o",
        action="store",
        help=(
            "Path to the output README file. If not provided, the generated content "
            "will be printed to standard output."
        ),
        metavar="<OUTPUT_README_FILE>",
    )

    # --- Environment Variable Override Arguments ---
    arg_env_var_group = argparser.add_argument_group(
        "Environment variables override",
        "These arguments can be used to override the default environment variables.",
    )
    arg_env_var_group.add_argument(
        "--gemini-api-key",
        "-api",
        default=os.environ.get("GH_SUMMARIZER_GEMINI_API_KEY"),
        help=(
            "Your Google Gemini API key. Overrides the "
            "`GH_SUMMARIZER_GEMINI_API_KEY` environment variable."
        ),
        metavar="<GEMINI_API_STRING>",
    )
    arg_env_var_group.add_argument(
        "--scp-user-prompt",
        "-scp",
        default=os.environ.get("GH_SUMMARIZER_SCP_USER_PROMPT"),
        help=(
            "The user prompt for summarizing SCP articles. Must contain `{scp_url}`. "
            "Overrides the `GH_SUMMARIZER_SCP_USER_PROMPT` environment variable."
        ),
        metavar="<SCP_USER_PROMPT>",
    )
    arg_env_var_group.add_argument(
        "--wikipedia-user-prompt",
        "-wiki",
        default=os.environ.get("GH_SUMMARIZER_WIKIPEDIA_USER_PROMPT"),
        help=(
            "The user prompt for summarizing Wikipedia articles. Must contain "
            "`{wikipedia_url}`. Overrides the `GH_SUMMARIZER_WIKIPEDIA_USER_PROMPT` "
            "environment variable."
        ),
        metavar="<WIKIPEDIA_USER_PROMPT>",
    )

    # Parse the command line arguments, using `argv` if provided
    args = argparser.parse_args(argv)

    # Assign parsed arguments to local variables for easier access
    max_retries = args.max_retries
    model = args.model

    scp_user_prompt = args.scp_user_prompt
    wikipedia_user_prompt = args.wikipedia_user_prompt
    gemini_api_key = args.gemini_api_key

    wikipedia_urls = args.wikipedia_urls
    readme_template = args.template
    output_readme = args.output_readme

    # Validate that all required configuration variables are present, either from
    # environment variables or command-line arguments
    if scp_user_prompt is None:
        error_details = (
            "GH_SUMMARIZER_SCP_USER_PROMPT",
            "SCP content summarization",
            "Please provide a correct and Python-formattable user prompt string.",
        )
        raise MissingEnvironmentVariableError(*error_details)

    if wikipedia_user_prompt is None:
        error_details = (
            "GH_SUMMARIZER_WIKIPEDIA_USER_PROMPT",
            "Wikipedia content summarization",
            "Please provide a correct and Python-formattable user prompt string.",
        )
        raise MissingEnvironmentVariableError(*error_details)

    if gemini_api_key is None:
        error_details = (
            "GH_SUMMARIZER_GEMINI_API_KEY",
            "content summarization",
            "Please provide a correct Gemini API key. "
            "You can get one from: https://aistudio.google.com/app/apikey",
        )
        raise MissingEnvironmentVariableError(*error_details)

    # Initialize the Google Gemini client with the provided API key
    gemini_client = ggenai.Client(api_key=gemini_api_key)

    # --- SCP Article Processing Loop ---
    # This loop attempts to fetch and summarize a random SCP article. It will continue
    # until a valid summary is obtained or a non-recoverable error occurs.
    scp_data = None
    while scp_data is None:
        # Attempt to fetch a random SCP article from the GraphQL API
        try:
            scp_raw_data = get_random_scp(max_retries=max_retries)
        except SCPResponseInvalidError:
            # Break the loop if the API consistently fails after retries
            break
        except Exception:
            # Log any other unexpected errors and exit the loop
            logger.exception("Encountered an unexpected error:")
            break

        # Extract the relevant page data from the nested GraphQL response
        scp_raw_page = scp_raw_data["data"]["randomPage"]["page"]
        scp_url = scp_raw_page["url"]

        # Attempt to summarize the fetched SCP article content
        try:
            scp_summary = summarize(
                gemini_client,
                scp_user_prompt.format(scp_url=scp_url),
                scp_system_prompt,
                model=model,
                max_retries=max_retries,
            )
        except (LLMResponseInappropriateError, LLMResponseTooShortError):
            # If the content is inappropriate or the summary is too short, try again
            # with a new random article
            continue
        except (LLMResponseInvalidError, LLMResponseError):
            # If the LLM response is invalid or there's a persistent error, break the
            # loop
            break
        except Exception:
            # Log any other unexpected errors and exit the loop
            logger.exception("Encountered an unexpected error:")
            break

        # If summarization is successful, create an `SCPData` object
        scp_data = SCPData(
            title=scp_raw_page["wikidotInfo"]["title"],
            title_alt=alt_titles[0]["title"]
            if (alt_titles := scp_raw_page["alternateTitles"])
            else "",
            url=scp_url,
            summary=scp_summary,
        )

    # --- Wikipedia Article Processing Loop ---
    # This loop follows the same logic as the SCP loop but for Wikipedia articles
    wikipedia_data = None
    while wikipedia_data is None:
        # Get a random Wikipedia article URL from the specified file
        wikipedia_url = get_random_wikipedia_url(wikipedia_urls_fp=wikipedia_urls)

        # Attempt to summarize the fetched Wikipedia article
        try:
            wikipedia_summary = summarize(
                gemini_client,
                wikipedia_user_prompt.format(wikipedia_url=wikipedia_url),
                wikipedia_system_prompt,
                model=model,
                max_retries=max_retries,
            )
        except (LLMResponseInappropriateError, LLMResponseTooShortError):
            # Retry with a new article if content is inappropriate or summary is too
            # short
            continue
        except (LLMResponseInvalidError, LLMResponseError):
            # If the LLM response is invalid or there's a persistent error, break the
            # loop
            break
        except Exception:
            # Log unexpected errors and exit
            logger.exception("Encountered an unexpected error:")
            break

        # If summarization is successful, create a `WikipediaData` object.
        wikipedia_data = WikipediaData(
            title=uparse.unquote(wikipedia_url.split("/")[-1]).replace("_", " "),
            url=wikipedia_url,
            summary=wikipedia_summary,
        )

    # Generate the final README content using the fetched data
    readme_text = generate_readme(
        scp_data,
        wikipedia_data,
        readme_fp=output_readme,
        readme_template_fp=readme_template,
    )

    # Print the generated README to standard output
    print(readme_text)


if __name__ == "__main__":
    # Replace the default logger with our custom one for direct script execution
    logger = _setup_logging(debug=True)

    # Run the main application logic
    _main()
