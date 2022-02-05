namespace AI.Net.Utilities;

public class Constants
{
    public static string SolutionPath()
    {
        return Directory.GetParent(AppDomain
                .CurrentDomain.BaseDirectory)?
            .Parent?.Parent?.Parent?.FullName;
    }
}