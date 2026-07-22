package com.example.imdbclone

/**
 * Minimal movie model draft.
 * Expand when wiring a real data source (TMDB/OMDb API, Room DB, etc.).
 */
data class Movie(
    val id: Int,
    val title: String,
    val year: Int? = null,
    val genres: List<String> = emptyList(),
    val moodTags: List<String> = emptyList(),
    val posterUrl: String? = null,
    val rating: Double? = null,
)
