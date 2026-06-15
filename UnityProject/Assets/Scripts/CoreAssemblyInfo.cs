using System.Runtime.CompilerServices;

// The pipeline pass Resource/Settings types in Core expose many `internal` fields
// that the feature assemblies (and their editors) populate when orchestrating passes.
// These were accessible when everything lived in Assembly-CSharp; after the assembly
// split we grant the feature assemblies access to Core internals explicitly.
[assembly: InternalsVisibleTo("PathTracing.NrdSample")]
[assembly: InternalsVisibleTo("PathTracing.Rtxdi")]
[assembly: InternalsVisibleTo("PathTracing.Rtxpt")]
