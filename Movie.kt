package com.example.imdbclone

data class Movie(
    val id: Int,
    val title: String,
    val year: String,
    val rating: Float,
    val genre: String,
    val description: String,
    val imageResId: Int  // We'll use local images for now
)
